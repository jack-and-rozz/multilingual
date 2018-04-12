#!/bin/bash
#
# usage: ./stanford-postagger.sh {textFile}

usage() {
    echo "Usage:$0 {input_file}"
    exit 1
}

if [ $# -lt 1 ];then
    usage;
fi


input_path=$1
ROOT_DIR=$STANFORD_POSTAGGER_ROOT
MODEL_PATH=$ROOT_DIR/models/english-left3words-distsim.tagger

if [ $# -lt 2 ];then
    memory=2g
else
    memory=$2
fi

 echo "java -mx$memory -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $input_path -outputFormat tsv -sentenceDelimiter newline -tokenize false > $input_path.pos 
"
java -mx$memory -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $input_path -outputFormat tsv -sentenceDelimiter newline -tokenize false > $input_path.pos 
