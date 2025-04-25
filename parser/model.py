
#Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#Written by Alireza Mohammadshahi <alireza.mohammadshahi@idiap.ch>,

#This file is part of g2g-transformer.

#g2g-transformer is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License version 2 as
#published by the Free Software Foundation.

#g2g-transformer is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with g2g-transformer. If not, see <http://www.gnu.org/licenses/>.


from parser.metric import Metric
import torch
import torch.nn as nn
import numpy as np
import numpy
from tqdm import tqdm
from parser.utils.mst import mst
import torch.nn.functional as F
from parser.contrast import Contrast
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer

class Model(object):

    def __init__(self, vocab, parser, config, num_labels):
        super(Model, self).__init__()

        self.vocab = vocab
        self.parser = parser
        self.num_labels = num_labels
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.emb_size = 768
        self.feat_dim = 128

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False, device='cuda:0'),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False, device='cuda:0'))
    
    def prepare_argmax(self, s_arc, s_rel, mask, sbert_arc, sbert_rel, stop_sign, just_pred, mask_gold=None):

        batch_size, max_len = mask.shape

        if mask_gold is not None:
            pred_arcs, pred_rels = s_arc[mask_gold], s_rel[mask_gold]
        else:
            pred_arcs, pred_rels = self.decode(s_arc.clone(), s_rel.clone(), mask.clone())
        
        matrix_arc = torch.zeros((batch_size,max_len)).long().to(mask.device)
        matrix_rel = torch.zeros((batch_size,max_len)).long().to(mask.device)
        
        lengths = mask.sum(dim=1)
        counter = 0
        for i,mask_instance in enumerate(mask):
            lens = len(mask_instance.nonzero())
            if lens != 0:
                matrix_arc[i,mask_instance] = pred_arcs[counter:counter+lens]
                matrix_rel[i,mask_instance] = pred_rels[counter:counter+lens]
                matrix_rel[i,1] = self.vocab.root_label
                counter += lens
                
        assert counter == len(pred_arcs)

        if self.config.subword_option == "syntax":
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_syn(matrix_arc,matrix_rel, mask.clone(), sbert_rel)
        else:
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_sem(matrix_arc, matrix_rel, mask.clone(), sbert_arc, sbert_rel)

        if just_pred:
            return pred_arcs, graph_rel

        if self.config.input_labeled_graph:
            graph_arc,pred_arcs, pred_rels, graph_rel = self.build_graph_labeled(pred_arcs,graph_rel,mask_new, stop_sign)
        else:
            graph_arc,pred_arcs, pred_rels, graph_rel = self.build_graph_unlabeled(pred_arcs,graph_rel,mask_new, stop_sign)

        if mask_gold is None:
            return graph_arc, graph_rel
        else:
            return graph_arc, pred_arcs, pred_rels, graph_rel

    def prepare_mst(self, s_arc, s_rel, mask, sbert_arc, sbert_rel, stop_sign, just_pred, mask_gold=None):

        if mask_gold is not None:
            batch_size, max_len = mask.shape
            p_arcs, pred_rels = s_arc[mask_gold], s_rel[mask_gold]
            pred_arcs = torch.zeros((batch_size, max_len)).long().to(mask.device)
            graph_rel = torch.zeros((batch_size, max_len)).long().to(mask.device)

            counter = 0
            for i, mask_instance in enumerate(mask):
                lens = len(mask_instance.nonzero())
                if lens != 0:
                    pred_arcs[i, mask_instance] = p_arcs[counter:counter + lens]
                    graph_rel[i, mask_instance] = pred_rels[counter:counter + lens]
                    graph_rel[i, 1] = self.vocab.root_label
                    counter += lens
        else:
            pred_arcs, graph_rel = self.decode_mst(s_arc.clone(), s_rel.clone(), mask.clone(), prepare=True)

        if self.config.subword_option == "syntax":
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_syn(pred_arcs,graph_rel, mask.clone(), sbert_rel)
        else:
            pred_arcs, graph_rel, mask_new = self.build_bert_graphs_sem(pred_arcs, graph_rel, mask.clone(), sbert_arc, sbert_rel)

        if just_pred:
            return pred_arcs,graph_rel

        if self.config.input_labeled_graph:
            graph_arc,pred_arcs, pred_rels, graph_rel = self.build_graph_labeled(pred_arcs,graph_rel,mask_new, stop_sign)
        else:
            graph_arc, pred_arcs, pred_rels, graph_rel = self.build_graph_unlabeled(pred_arcs,graph_rel,mask_new, stop_sign)

        if mask_gold is None:
            return graph_arc, graph_rel
        else:
            return graph_arc, pred_arcs, pred_rels, graph_rel

    def build_bert_graphs_sem(self, pred_arcs, pred_rels, mask, sbert_arc, sbert_rel):

        sbert_arc[mask] = pred_arcs[mask]
        sbert_arc[:,1] = 0
        mask_new = sbert_rel == self.vocab.sbert_label
        mask_new[:,1] = False

        sbert_rel[mask] = pred_rels[mask]
        sbert_rel[:,1] = self.vocab.root_label

        return sbert_arc, sbert_rel, mask_new

    def build_bert_graphs_syn(self, pred_arcs, pred_rels, mask, sbert_rel):

        mask_new = sbert_rel == self.vocab.sbert_label
        mask_new[:, 1] = False
        mask_total = mask_new.long() + mask.long()

        new_graphs_arc = torch.zeros(mask_total.shape[0],mask_total.shape[1]).long().to(pred_arcs.device)
        new_graphs_rel = torch.zeros(mask_total.shape[0],mask_total.shape[1]).long().to(pred_arcs.device)
        for i,(pred_arc,pred_rel,mask_instance) in enumerate(zip(pred_arcs,pred_rels,mask_total)):
            for j,m in enumerate(mask_instance):
                if m==2:
                    new_graphs_arc[i,j] = pred_arc[j]
                    new_graphs_rel[i,j] = pred_rel[j]
                elif m==1:
                    new_graphs_arc[i,j] = new_graphs_arc[i][j-1]
                    new_graphs_rel[i,j] = new_graphs_rel[i][j-1]
        return new_graphs_arc, new_graphs_rel, mask_new

    def build_graph_unlabeled(self, pred_arcs, graph_rel, mask, stop_sign):
    
        graph_arc = torch.zeros(mask.shape[0],mask.shape[1],mask.shape[1]).long().to(mask.device)
        
        mask = mask.long()
        
        mask = stop_sign.unsqueeze(1) * mask
        
        lengths = mask.sum(dim=1)
        
        graph_rel = graph_rel * mask

        for i,(arc,lens,mask_instance) in enumerate(zip(pred_arcs,lengths,mask)):
            
            if lens != 0:
                graph_arc[i,torch.arange(mask.shape[1]),arc] = 1
                graph_arc[i,:,:] = graph_arc[i,:,:] * mask[i].unsqueeze(1)
                graph_arc[i,:,:] = graph_arc[i,:,:] + 2 * graph_arc[i,:,:].transpose(0,1)

                if not self.config.use_mst_train:
                    graph_arc[i,:,:] = graph_arc[i,:,:] * (graph_arc[i,:,:] != 3)
                assert not len( (graph_arc[i,:,:] == 3).nonzero() )
            if self.config.train_zero_rel:
                mask_instance[1] = 1
                mask_new = (mask_instance.unsqueeze(0) * mask_instance.unsqueeze(1)).bool()
                ## 3 is padding_idx
                graph_arc[i,:,:][~mask_new] = 3

        return graph_arc, pred_arcs, graph_rel, graph_rel
    
    def build_graph_labeled(self, pred_arcs, pred_rel, mask, stop_sign):

        graph_arc = torch.zeros(mask.shape[0],mask.shape[1],mask.shape[1]).long().to(mask.device)
        mask = mask.long()
        
        mask = stop_sign.unsqueeze(1) * mask
        
        lengths = mask.sum(dim=1)
        
        for i,(arc,rel,lens, mask_instance) in enumerate(zip(pred_arcs,pred_rel,lengths,mask)):
            
            if lens != 0:
                graph_arc[i,torch.arange(mask.shape[1]),arc] = rel + 1
                graph_arc[i,:,:] = graph_arc[i,:,:] * mask[i].unsqueeze(1)
            
                assert not( len( (graph_arc[i,:,:] < 0).nonzero() ) or len( (graph_arc[i,:,:] > 
                                                                                   self.num_labels).nonzero() ) ) 
                mask_t = (graph_arc[i,:,:] > 0)*1
                graph_arc_t = graph_arc[i,:,:].transpose(0,1) + mask_t.transpose(0,1) * self.num_labels
                graph_arc[i,:,:] = graph_arc[i,:,:] + graph_arc_t
            
                if not self.config.use_mst_train:
                    mask_t = mask_t + mask_t.transpose(0,1)
                    mask_t = mask_t * (mask_t != 2).long() 
                    graph_arc[i,:,:] = graph_arc[i,:,:] * mask_t

                assert not len( (graph_arc[i,:,:] > 2*self.num_labels).nonzero() )

            if self.config.train_zero_rel:
                mask_instance[1] = 1
                mask_new = (mask_instance.unsqueeze(0) * mask_instance.unsqueeze(1)).bool()
                graph_arc[i,:,:][~mask_new] = 2 * self.num_labels + 1
                

        return graph_arc, pred_arcs, pred_rel, None    
    
    def check_stop(self, stop_sign, arc_new, arc_prev, rel_new, rel_prev, mask):
        
        for i,(mask_instance,narc,parc,nrel,prel) in enumerate(zip(mask,arc_new,arc_prev,rel_new,rel_prev)):
            if len(mask_instance.nonzero()) != 0:
                dif_arc = len( (narc[mask_instance]-parc[mask_instance]).nonzero() ) 
                dif_rel = len( (nrel[mask_instance]-prel[mask_instance]).nonzero() )
                if self.config.stop_arc and dif_arc==0:
                    stop_sign[i] = 0
                elif self.config.stop_arc_rel and dif_arc==0 and dif_rel==0:
                    stop_sign[i] = 0
        
        return stop_sign

    def train_predicted(self, loader):
        self.parser.train()
        pbar = tqdm(total=len(loader))

        pos_loader = Contrast(self.config)
        
        for anchor, positive in zip(loader, pos_loader):
            words, tags, arcs, rels, initial_heads, initial_rels, mask, sbert_arc, sbert_rel = anchor
            pos_words, pos_tags, pos_arcs, pos_rels, pos_initial_heads, pos_initial_rels, pos_mask, pos_sbert_arc, pos_sbert_rel = pos_loader

            mask_gold = arcs > 0
            stop_sign = torch.ones(len(words)).long().to(words.device)

            ## iterate over encoder
            for counter in range(1,self.config.num_iter_encoder):
                if counter == 1:
                    graph_arc, prev_arcs, prev_rels, graph_rel = \
                            self.prepare_mst(initial_heads, initial_rels, mask, sbert_arc.clone(),
                                             sbert_rel.clone(), stop_sign,False,mask_gold)
                    pos_graph_arc, pos_prev_arcs, pos_prev_rels, pos_graph_rel = \
                            self.prepare_mst(pos_initial_heads, pos_initial_rels, pos_mask, pos_sbert_arc.clone(),
                                             pos_sbert_rel.clone(), stop_sign,False,mask_gold)
                else:
                    if self.config.use_mst_train:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                        pos_graph_arc,pos_graph_rel= \
                        self.prepare_mst(pos_s_arc,pos_s_rel,pos_mask,pos_sbert_arc.clone(),pos_sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                        pos_graph_arc,pos_graph_rel= \
                        self.prepare_argmax(pos_s_arc,pos_s_rel,pos_mask,pos_sbert_arc.clone(),pos_sbert_rel.clone(),stop_sign,False)

                x, s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)
                pos_x, pos_s_arc, pos_s_rel = self.parser(pos_words, pos_tags, stop_sign, pos_graph_arc, pos_graph_rel)
                s_arc_t = s_arc[mask]
                s_rel_t = s_rel[mask, :]
                pos_s_arc_t = pos_s_arc[pos_mask]
                pos_s_rel_t = pos_s_rel[pos_mask:]
                gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

                if self.config.use_two_opts:
                    self.optimizer_nonbert.zero_grad()
                    self.optimizer_bert.zero_grad()
                else:
                    self.optimizer.zero_grad()


                ce_loss = self.get_loss(s_arc_t, s_rel_t, gold_arcs, gold_rels)
                loss_func = losses.SelfSupervisedLoss(losses.SupConLoss())
                # hard_pairs = miner(s_arc_t, gold_arcs)
                cl_loss = loss_func(s_arc_t, pos_s_arc_t)
                # print(f"ce_loss : {ce_loss}, cl_loss : {cl_loss}")
                loss = ce_loss + cl_loss
                loss.backward()

                if self.config.use_two_opts:
                    self.optimizer_nonbert.step()
                    self.optimizer_bert.step()
                    self.scheduler_nonbert.step()
                    self.scheduler_bert.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()


                if self.config.use_mst_train or counter==1:
                    new_arcs,new_rels = self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),
                                                               sbert_rel.clone(),stop_sign,True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),
                                                                         sbert_rel.clone(),stop_sign,True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        #print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()
                    mask_gold = (stop_sign.unsqueeze(1) * mask_gold.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels


            pbar.update(1)


    def train(self, loader):
        self.parser.train()
        pbar = tqdm(total= len(loader))

        pos_loader = Contrast(self.config)

        for anchor, positive in zip(loader, pos_loader):

            words, tags, arcs, rels, mask, sbert_arc, sbert_rel = anchor
            pos_words, pos_tags, pos_arcs, pos_rels, pos_mask, pos_sbert_arc, pos_sbert_rel = positive
            # neg_words, neg_tags, neg_arcs, neg_rels, neg_mask, neg_sbert_arc, neg_sbert_rel = negative

            # print(f"Shape of words : {words.size()}")
            # print(f"Shape of arcs : {arcs.size()}")
            # # print(f"Words : {words}")
            # print(f"word : {words[0]} and shape of each word : {words[0].size()}")
            # print(f"arc : {arcs[0]} and shape of each arc : {arcs[0].size()}")
            # print(f"mask : {mask}") 
            # print(f"Shape of Positive Words : {pos_words.size()}") 
            # print(f"positive words : {pos_words}")    
            mask_gold = arcs > 0
            stop_sign = torch.ones(len(words)).long().to(words.device) 
            pos_mask_gold = pos_arcs > 0 # needed for CE loss
            pos_stop_sign = torch.ones(len(pos_words)).long().to(pos_words.device) 

            ## iterate over encoder
            cl_loss = 0.0
            for counter in range(self.config.num_iter_encoder):
                if counter==0:
                    x, s_arc, s_rel = self.parser(words, tags)
                    pos_x, pos_s_arc, pos_s_rel = self.parser(pos_words, pos_tags)
                    # neg_s_arc, neg_s_rel = self.parser(neg_words, neg_tags)
                else:
                    if self.config.use_mst_train or counter==1:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                        pos_graph_arc,pos_graph_rel= \
                        self.prepare_mst(pos_s_arc,pos_s_rel,pos_mask,pos_sbert_arc.clone(),pos_sbert_rel.clone(),pos_stop_sign,False)
                        # neg_graph_arc,neg_graph_rel= \
                        # self.prepare_mst(neg_s_arc,neg_s_rel,neg_mask,neg_sbert_arc.clone(),neg_sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                        pos_graph_arc, pos_graph_rel = \
                        self.prepare_argmax(pos_s_arc,pos_s_rel,pos_mask,pos_sbert_arc.clone(),pos_sbert_rel.clone(),pos_stop_sign,False)
                        # neg_graph_arc,neg_graph_rel= \
                        # self.prepare_argmax(neg_s_arc,neg_s_rel,neg_mask,neg_sbert_arc.clone(),neg_sbert_rel.clone(),stop_sign,False)
                    # print(f"Shape of graph_arc : {graph_arc.size()}")
                    x, s_arc,s_rel = self.parser(words,tags,stop_sign,graph_arc,graph_rel)
                    pos_x , pos_s_arc, pos_s_rel = self.parser(pos_words, pos_tags, pos_graph_arc, pos_graph_rel)
                    # neg_s_arc, neg_s_rel = self.parser(neg_words, neg_tags, stop_sign, neg_graph_arc, neg_graph_rel)
                
                # pos_x = pos_x.detach()
                # print(f"Shape of s_arc : {s_arc.size()}")
                # # print(f"Shape of s_rel : {s_rel.size()}")
                # print(f"Shape of pos_s_arc : {pos_s_arc.size()}")
                # # print(f"Shape of pos_s_rel : {pos_s_rel.size()}")
                # print(f"Shape of mask : {mask.size()}")
                
                # print(f"s_arc : {s_arc}")
                # print(f"mask : {mask}")
                s_arc_t = s_arc[mask]
                s_rel_t = s_rel[mask,:]


                pos_s_arc_t = pos_s_arc[pos_mask]
                pos_s_rel_t = pos_s_rel[pos_mask,:]

                # neg_s_arc_t = neg_s_arc[neg_mask]
                # neg_s_rel_t = neg_s_rel[neg_mask,:]
                gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]
                pos_gold_arcs, pos_gold_rels = pos_arcs[pos_mask_gold], pos_rels[pos_mask_gold]

                if self.config.use_two_opts:
                    self.optimizer_nonbert.zero_grad()
                    self.optimizer_bert.zero_grad()
                else:
                    self.optimizer.zero_grad()
                
                # print(f"In Loop : {counter+1}")
                # print(f"Shape of s_arc_t : {s_arc_t.size()}")
                # print(f"Shape of mask : {mask.size()}")
                # # print(f"Shape of s_rel_t : {s_rel_t.size()}")
                # print(f"Shape of pos_arc_t : {pos_s_arc_t.size()}")
                # print(f"Shape of pos_mask : {pos_mask.size()}")
                # print(f"Shape of pos_rel_t : {pos_s_rel_t.size()}")
                # print(f"Shape of neg_arc_t : {neg_s_arc_t.size()}")
                # print(f"Shape of neg_rel_t : {neg_s_rel_t.size()}")
                # print(f"s_arc_t : {s_arc_t}")
                # print(f"pos_s_arc_t : {pos_s_arc_t}")

                # compute mle loss
                ce_loss = self.get_loss(s_arc_t, s_rel_t, gold_arcs, gold_rels)

                # prepare and compute the contrastive loss 
                # taking the pooled output from bert and getting the mean embeddings 
                # using attention mask from bert to tell the first head words and not take the other subwords
                mean_out_1 = self.get_mean_embeddings(x, mask)
                mean_out_2 = self.get_mean_embeddings(pos_x, pos_mask)

                # print(f"Shape of mean_out_1 : {mean_out_1.size()}")
                # print(f"Shape of mean_out_2 : {mean_out_2.size()}")

                cnst_feat_1, cnst_feat_2 = self.contrast_logits(mean_out_1.to('cuda:0'), mean_out_2.to('cuda:0'))
                # # print(f"Shape of tensor : {mean_out_1}, shape of aug tensor : {mean_out_2}")
                cl_loss = self.contrastive_loss(cnst_feat_1, cnst_feat_2)
                # loss_func = losses.SelfSupervisedLoss(losses.SupConLoss())
                # cl_loss = loss_func(s_arc_t, pos_s_arc_t)
                
                # print(f"ce_loss : {ce_loss}, cl_loss : {cl_loss}")
                loss = ce_loss + cl_loss
                loss.backward()

                if self.config.use_two_opts:
                    self.optimizer_nonbert.step()
                    self.optimizer_bert.step()
                    self.scheduler_nonbert.step()
                    self.scheduler_bert.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()


                if self.config.use_mst_train or counter==2:
                    new_arcs,new_rels = self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),
                                                               sbert_rel.clone(),stop_sign,True)
                    pos_new_arcs,pos_new_rels = self.prepare_mst(pos_s_arc,pos_s_rel,pos_mask,pos_sbert_arc.clone(),
                                                               pos_sbert_rel.clone(),pos_stop_sign,True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),
                                                                         sbert_rel.clone(),stop_sign,True)
                    pos_new_arcs,pos_new_rels = self.prepare_argmax(pos_s_arc,pos_s_rel,pos_mask,pos_sbert_arc.clone(),
                                                               pos_sbert_rel.clone(),pos_stop_sign,True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    pos_stop_sign = self.check_stop(pos_stop_sign, pos_new_arcs, pos_prev_arcs, pos_new_rels, pos_prev_rels, pos_mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    # print(f"Shape of stop_sign in loop {counter+1} : {stop_sign.size()}")
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()
                    mask_gold = (stop_sign.unsqueeze(1) * mask_gold.long()).bool()

                    # print(f"Shape of mask in loop {counter + 1} : {mask.size()}")

                    pos_mask = (pos_stop_sign.unsqueeze(1) * pos_mask.long()).bool()
                    pos_mask_gold = (pos_stop_sign.unsqueeze(1) * pos_mask_gold.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                pos_prev_arcs = pos_new_arcs
                pos_prev_rels = pos_new_rels

                # print(f"Shape of new_arcs : {new_arcs.size()}")

            pbar.update(1)

    @torch.no_grad()
    def evaluate_predicted(self, loader, punct=False):
        self.parser.eval()

        loss, metric = 0, Metric()
        pbar = tqdm(total=len(loader))
        for words, tags, arcs, rels, initial_heads, initial_rels, mask, sbert_arc, sbert_rel in loader:

            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs > 0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(1,self.config.num_iter_encoder):

                self.counter_ref = counter

                if counter == 1:
                    graph_arc, prev_arcs, prev_rels, graph_rel = \
                            self.prepare_mst(initial_heads, initial_rels, mask, sbert_arc.clone(),
                                             sbert_rel.clone(), stop_sign,False,mask_gold)
                else:
                    if self.config.use_mst_train:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)


                x, s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)

                if self.config.use_mst_train or counter == 1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 1:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                if counter == 1:
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                else:
                    index = stop_sign.nonzero().squeeze(1)
                    s_arc_final[index] = s_arc[index]
                    s_rel_final[index] = s_rel[index]

            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]
            if self.config.use_mst_eval:
                pred_arcs, pred_rels = self.decode_mst(s_arc_final,s_rel_final, mask_unused,prepare=False)
            else:
                pred_arcs, pred_rels = self.decode(s_arc_final, s_rel_final, mask_unused)

            s_arc_mask = s_arc_final[mask_unused]
            s_rel_mask = s_rel_final[mask_unused]

            loss += self.get_loss(s_arc_mask, s_rel_mask, gold_arcs, gold_rels)

            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
            pbar.update(1)

        loss /= len(loader)
        return loss, metric


    @torch.no_grad()
    def evaluate(self, loader, punct=False):
        self.parser.eval()

        loss, metric = 0, Metric()
        pbar = tqdm(total=len(loader))
        for words, tags, arcs, rels, mask, sbert_arc, sbert_rel in loader:
            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs>0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(self.config.num_iter_encoder):

                self.counter_ref = counter

                if counter==0:
                    x, s_arc, s_rel = self.parser(words, tags)
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                    # print(f"s_arc from evaluate : {s_arc}")
                else:
                    if self.config.use_mst_train or counter==1:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)

                    x, s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)


                if self.config.use_mst_train or counter==1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                index = stop_sign.nonzero()
                s_arc_final[index] = s_arc[index]
                s_rel_final[index] = s_rel[index]

            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

            # print(f"s_arc_final : {s_arc_final}")

            if self.config.use_mst_eval:
                pred_arcs, pred_rels = self.decode_mst(s_arc_final,s_rel_final, mask_unused,prepare=False)
            else:
                pred_arcs, pred_rels = self.decode(s_arc_final, s_rel_final,mask_unused)


            s_arc_mask = s_arc_final[mask_unused]
            s_rel_mask = s_rel_final[mask_unused]
            
            # compute cl loss
            # norm_rep_arc = s_arc / s_arc.norm(dim=2, keepdim=True)
            # cosine_scores_arc = torch.matmul(norm_rep_arc, norm_rep_arc.transpose(1,2))

            # print(f"Shape of score_arc : {cosine_scores_arc.size()}")

            # norm_rep_rel = s_rel / s_rel.norm(dim=3, keepdim=True)
            # print(f"Shape of norm_rep_rel : {norm_rep_rel.size()}")
            # cosine_scores_rel = torch.matmul(norm_rep_rel, norm_rep_rel.transpose(2,3))

            # print(f"Shape of score_rel : {cosine_scores_rel.size()}")

            # cl_loss = self.contrastive_loss_arc(cosine_scores_arc, words) # + self.contrastive_loss_rel(cosine_scores_rel, tags)
            # miner = miners.MultiSimilarityMiner()
            loss_func = losses.SelfSupervisedLoss(losses.SupConLoss())

            # hard_pairs = miner(s_arc_mask, gold_arcs)
            # cl_loss = loss_func(s_arc_mask, gold_arcs)
            ce_loss = self.get_loss(s_arc_mask, s_rel_mask, gold_arcs, gold_rels)            
            loss = ce_loss # + cl_loss
            
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
            pbar.update(1)

        loss /= len(loader)
        return loss, metric
    


    @torch.no_grad()
    def predict_predicted(self, loader):
        self.parser.eval()
        metric = Metric()
        all_arcs, all_rels = [], []


        for words, tags, arcs, rels, initial_heads, initial_rels, mask, sbert_arc, sbert_rel in loader:

            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs > 0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(1,self.config.num_iter_encoder):
                if counter == 1:
                    graph_arc, prev_arcs, prev_rels, graph_rel = \
                            self.prepare_mst(initial_heads, initial_rels, mask, sbert_arc.clone(),
                                             sbert_rel.clone(), stop_sign,False,mask_gold)
                else:
                    if self.config.use_mst_train:
                        graph_arc,graph_rel= \
                        self.prepare_mst(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)
                    else:
                        graph_arc,graph_rel= \
                        self.prepare_argmax(s_arc,s_rel,mask,sbert_arc.clone(),sbert_rel.clone(),stop_sign,False)


                x, s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)

                if self.config.use_mst_train or counter==1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 1:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                if counter == 1:
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                else:
                    index = stop_sign.nonzero().squeeze(1)
                    s_arc_final[index] = s_arc[index]
                    s_rel_final[index] = s_rel[index]



            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

            if self.config.use_mst_eval:
                pred_rels, pred_arcs_org, pred_arcs = self.decode_mst(s_arc_final, s_rel_final,
                                                           mask_unused,prepare=False,do_predict=True)


            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

            lens = mask_unused.sum(1).tolist()

            all_arcs.extend(torch.split(pred_arcs_org, lens))
            all_rels.extend(torch.split(pred_rels, lens))

        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels, metric


    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        metric = Metric()
        all_arcs, all_rels = [], []
        for words, tags, arcs, rels, mask, sbert_arc, sbert_rel, offsets in loader:

            stop_sign = torch.ones(len(words)).long().to(words.device)
            mask_gold = arcs > 0
            mask_unused = mask.clone()
            ## iterate over encoder
            for counter in range(0,self.config.num_iter_encoder):

                self.counter_ref = counter

                if counter == 0:
                    x, s_arc, s_rel = self.parser(words, tags)
                    s_arc_final = s_arc
                    s_rel_final = s_rel
                else:
                    if self.config.use_mst_train or counter==1:
                        graph_arc, graph_rel = \
                            self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(), sbert_rel.clone(), stop_sign, False)
                    else:
                        graph_arc, graph_rel = \
                            self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(), sbert_rel.clone(), stop_sign,
                                                False)
                    x, s_arc, s_rel = self.parser(words, tags, stop_sign, graph_arc, graph_rel)

                if self.config.use_mst_train or counter==1:
                    new_arcs, new_rels = self.prepare_mst(s_arc, s_rel, mask, sbert_arc.clone(),
                                                          sbert_rel.clone(), stop_sign, True)
                else:
                    new_arcs, new_rels = self.prepare_argmax(s_arc, s_rel, mask, sbert_arc.clone(),
                                                             sbert_rel.clone(), stop_sign, True)

                if counter > 0:
                    stop_sign = self.check_stop(stop_sign, new_arcs, prev_arcs, new_rels, prev_rels, mask)
                    if stop_sign.sum() == 0:
                        # print('All Dependency Graphs are converged in this batch')
                        break
                    mask = (stop_sign.unsqueeze(1) * mask.long()).bool()

                prev_arcs = new_arcs
                prev_rels = new_rels

                index = stop_sign.nonzero()
                s_arc_final[index] = s_arc[index]
                s_rel_final[index] = s_rel[index]

            gold_arcs, gold_rels = arcs[mask_gold], rels[mask_gold]

            if self.config.use_mst_eval:
                pred_rels, pred_arcs_org,pred_arcs = self.decode_mst(s_arc_final, s_rel_final,
                                                           mask_unused,prepare=False,do_predict=True)


            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

            lens = mask_unused.sum(1).tolist()

            all_arcs.extend(torch.split(pred_arcs_org, lens))
            all_rels.extend(torch.split(pred_rels, lens))

        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels, metric

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)

        loss = arc_loss + rel_loss

        return loss
    
    # the part for contrastive loss

    def contrastive_loss(self, features_1, features_2, temperature=0.05):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2,2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1*features_2, dim=-1)/ temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous())/temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1) 

        Ng = neg.sum(dim=-1)
        loss = -torch.log(pos/(Ng+pos)).mean()
        return loss
    
    def get_mean_embeddings(self, word_emb, attention_mask):
        # bert_output = self.roberta.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(word_emb*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output
     
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(embd1.to('cuda:0'), dim=1).to('cuda:0')
        if embd2 != None:
            feat2 = F.normalize(embd2.to('cuda:0'), dim=1).to('cuda:0')
            return feat1, feat2
        else: 
            return feat1
    
    def decode(self, s_arc, s_rel, mask):
        
        mask_new = self._mask_arc(s_arc,mask.clone())
        
        s_arc = s_arc + (1. - mask_new) * (-1e8)
        
        s_arc = F.softmax(s_arc,dim=2)        
         
        s_arc = s_arc * mask_new
        
        
        s_arc = s_arc[mask]
        s_rel = s_rel[mask]

        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels

    def re_map(self,pred_arc,mask,do_predict=False):

        dict = {-1:0}
        indicies = mask.nonzero()
        for counter, index in enumerate(indicies):
            dict[counter] = int(index)
        pred_arc_org = np.zeros(len(mask))

        if not do_predict:
            for i in range(len(pred_arc)):
                pred_arc[i] = dict[pred_arc[i]]

        pred_arc_org[mask.bool().cpu().numpy()] = pred_arc

        pred_arc_org *= mask.bool().cpu().numpy()

        return pred_arc_org
        
    def _mask_arc(self, logits_arc, all_mask):
        mask_new = torch.zeros(logits_arc.shape).to(logits_arc.device)
        self_loop = (1-torch.eye(logits_arc.shape[2])).to(logits_arc.device)
        all_mask = all_mask.long()
        for i, mask in enumerate(all_mask):
            mask_new[i,:,:] = mask.unsqueeze(0) * mask.unsqueeze(1)
            mask_new[i,:,1] = 1
            mask_new[i,:,:] *= self_loop
                
        return mask_new.to(logits_arc.device)  

    def decode_mst(self, s_arc, s_rel, mask, prepare, do_predict=False):
        
        mask_new = self._mask_arc(s_arc,mask.clone())
        
        s_arc = s_arc + (1. - mask_new) * (-1e8)

        s_arc = F.softmax(s_arc,dim=2)
        
        s_arc = s_arc*mask_new
        
        
        s_arc_final = np.zeros((mask.shape[0],mask.shape[1]))
        if do_predict:
            s_arc_final_org = np.zeros((mask.shape[0],mask.shape[1]))
        s_rel_final = torch.zeros((mask.shape[0],mask.shape[1])).long().to(mask.device)
        
        for counter, (s_arc_batch,s_rel_batch,mask_instance) in enumerate(zip(s_arc,s_rel,mask)):
            
            if len(mask_instance.nonzero())!= 0:
                
                ## set root mask True
                mask_instance[1] = True
                ## predict the dependencies on word level
                s_arc_batch = s_arc_batch[mask_instance.unsqueeze(0) * mask_instance.unsqueeze(1)].reshape(mask_instance.sum(),mask_instance.sum())
                
                if prepare:
                    pred_arc,_ = mst(s_arc_batch.cpu().detach().numpy(), use_chi_liu_edmonds=True)
                else:
                    pred_arc,_ = mst(s_arc_batch.cpu().numpy(), use_chi_liu_edmonds=True)
               
                
                ## covnert back to original positions
                if do_predict:
                    pred_arc_org = self.re_map(pred_arc, mask_instance.clone(),True)

                pred_arc = self.re_map(pred_arc,mask_instance.clone())


                s_arc_final[counter] = pred_arc

                if do_predict:
                    s_arc_final_org[counter] = pred_arc_org

                s_rel_batch = s_rel_batch.clone()
                s_rel_batch[:,:,self.vocab.root_label] = -1e8
                
                ## predict labels
                pred_rels = s_rel_batch[torch.arange(len(pred_arc)),pred_arc].argmax(dim=-1)
            
                s_rel_final[counter] = pred_rels
                s_rel_final[counter,1] = self.vocab.root_label
                #set root mask False
                mask_instance[1] = False                

            
            
        s_arc_final = torch.from_numpy(s_arc_final).long().to(mask.device)
        if do_predict:
            s_arc_final_org = torch.from_numpy(s_arc_final_org).long().to(mask.device)

        if prepare:
            return s_arc_final, s_rel_final
        else:
            if do_predict:
                return s_rel_final[mask], s_arc_final_org[mask], s_arc_final[mask]
            else:
                return s_arc_final[mask], s_rel_final[mask]
            
        

        
        
        
      
           
        
        
