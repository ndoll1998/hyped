EXP_NAME=imdb_bert
# paths to configurations
DATA_CONF=configs/${EXP_NAME}_data.json
EXP_CONF=configs/${EXP_NAME}_exp.json
# path to output directory
OUTPUT_DIR=output/${EXP_NAME}
# keep datasets up to 10GB in memory
export HF_DATASETS_IN_MEMORY_MAX_SIZE=1e10
#export CUDA_VISIBLE_DEVICES=0 # multi-heads currently only support single gpu

# data preparation stage
# runs the data preparation pipeline specified in the
# configuration file and saves the prepared dataset
# to the specified location
#python -m hyped.stages.prepare \
#    -c $DATA_CONF \
#    -o $OUTPUT_DIR/data

# model training stage
# trains a model on the prepared data generated by the
# previous stage
python -m hyped.stages.train \
    -c $EXP_CONF \
    -d $OUTPUT_DIR/data \
    -o $OUTPUT_DIR/model
exit
# model evaluation stage
# evaluates the trained model on the test split of the
# prepared dataset generated earlier
python -m hyped.stages.test \
    -c $EXP_CONF \
    -d $OUTPUT_DIR/data \
    -m $OUTPUT_DIR/model/best-model
