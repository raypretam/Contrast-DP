#! /bin/bash
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Alireza Mohammadshahi <alireza.mohammadshahi@idiap.ch>,

# This file is part of g2g-transformer.

# g2g-transformer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.

# g2g-transformer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with g2g-transformer. If not, see <http://www.gnu.org/licenses/>.

MODELPATH="/home/preetam_pg/g2g-transformer/models/STBC"
TRAIN_PATH="/home/preetam_pg/g2g-transformer/data/STBC/train.conllu"
POS_PATH="/home/preetam_pg/g2g-transformer/data/STBC/train.conllu"
DEV_PATH="/home/preetam_pg/g2g-transformer/data/STBC/dev.conllu"
TEST_PATH="/home/preetam_pg/g2g-transformer/data/sishu/poetry.conllu"
BERT_PATH="/home/preetam_pg/g2g-transformer/mbert"
# output of initial parser
INIT_TRAIN_PATH="/home/preetam_pg/g2g-transformer/pred/STBC/train.conllu"
INIT_TEST_PATH="/home/preetam_pg/g2g-transformer/pred/STBC/test.conllu"
INIT_POS_PATH="/home/preetam_pg/g2g-transformer/pred/STBC/train.conllu"
INIT_DEV_PATH="/home/preetam_pg/g2g-transformer/pred/STBC/dev.conllu"
#conllu or conllx
INPUT_TYPE="conllu"
python run.py train --lr1 1e-5 --lr2 2e-3 -w 0.001 \
                    --modelpath $MODELPATH --num_iter_encoder 6 --batch_size 16 --ftrain $TRAIN_PATH --ftest $TEST_PATH --fdev $DEV_PATH \
                    --punct --bert_path $BERT_PATH --input_labeled_graph --use_mst_eval --use_two_opts --layernorm_key \
                    --fpredicted_train $INIT_TRAIN_PATH --fpredicted_dev  $INIT_DEV_PATH --fpredicted_test  $INIT_TEST_PATH\
                    --use_predicted --stop_arc_rel --input_type $INPUT_TYPE --fpredicted_pos $INIT_POS_PATH
