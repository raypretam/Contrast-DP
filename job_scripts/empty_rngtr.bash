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

MODELPATH="/home/preetamray-pg/g2g-transformer/models/turkish"
TRAIN_PATH="/home/preetamray-pg/g2g-transformer/data/UD_Turkish-Penn/tr_penn-ud-train.conllu"
POS_PATH="/home/preetamray-pg/g2g-transformer/data/UD_Turkish-Penn/tr_penn-ud-train-pos.conllu"
DEV_PATH="/home/preetamray-pg/g2g-transformer/data/UD_Turkish-Penn/tr_penn-ud-dev.conllu"
TEST_PATH="/home/preetamray-pg/g2g-transformer/data/UD_Turkish-Penn/tr_penn-ud-test.conllu"
BERT_PATH="/home/preetamray-pg/g2g-transformer/mbert"
INPUT_TYPE="conllu"
python run.py train --lr1 1e-5 --lr2 2e-3 -w 0.001 \
                    --modelpath $MODELPATH --num_iter_encoder 2 --batch_size 8 \
                    --ftrain $TRAIN_PATH --ftest $TEST_PATH --fdev $DEV_PATH --punct --bert_path $BERT_PATH \
                    --input_labeled_graph --use_mst_eval --stop_arc_rel --use_two_opts --layernorm_key --same_flag \
                    --input_type $INPUT_TYPE --positive $POS_PATH





