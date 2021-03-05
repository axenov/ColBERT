train_data="eval.multi.triples.train.small.tsv"
batch=100
dim=64
model=base
experiment=base
steps=400000

while getopts f:n:b:d:m:e: flag
do
    case "${flag}" in
        f) train_data=${OPTARG};;
        n) steps=${OPTARG};;
        b) batch=${OPTARG};;
        d) dim=${OPTARG};;
		m) model=${OPTARG};;
		e) experiment=${OPTARG};;
    esac
done

if [[ $model = "large" ]]
then
  git checkout 0.2_large_cloud
fi

if [[ $model = "base" ]]
then
  git checkout 0.2_cloud
fi

mkdir -p DATA

if test ! -f "DATA/${train_data}"; then
    aws s3 cp "s3://datasets/german_ms_marco/${train_data}.zip" ./DATA --endpoint-url $AWS_ENDPOINT_URL
    unzip "DATA/${train_data}.zip" -d "DATA/"
    rm "DATA/${train_data}.zip"

fi


python -m colbert.train --triples DATA/${train_data} --query_maxlen 32 --doc_maxlen 150 --bsize $batch --dim $dim --amp --experiment $experiment --maxsteps $steps
