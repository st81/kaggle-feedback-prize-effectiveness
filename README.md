# Setup GCP instance

Not explicitly specified, commands must be run on gcp instance.

## Instance config

- us-central1-c
- A100 40GB

## Send required files

**Run below command in local**

```sh
PROJECT="sigma-night-266302"
REGION="us-central1-c"
INSTANCE_NAME="instance-1"
gcloud compute scp  --zone $REGION --project $PROJECT --recurse docker $INSTANCE_NAME:~
gcloud compute scp  --zone $REGION --project $PROJECT --recurse src/requirements.txt $INSTANCE_NAME:~
gcloud compute scp  --zone $REGION --project $PROJECT --recurse /mnt/c/Users/m59rt/.kaggle/kaggle.json $INSTANCE_NAME:~
```

## Setup directories and files

```sh
sudo mkdir -p /root/.kaggle
sudo cp ~/kaggle.json /root/.kaggle
sudo chmod 600 /root/.kaggle/kaggle.json

sudo mkdir -p /kaggle/input /kaggle/working
```

## Download required files from kaggle

```sh
dataset="feedback-prize-effectiveness"
kaggle competitions download -c $dataset -p ../input
mkdir ../input/$dataset
unzip -o ../input/$dataset.zip -d ../input/$dataset

dataset="feedback-prize-2021"
kaggle competitions download -c $dataset -p ../input
mkdir ../input/$dataset
unzip -o ../input/$dataset.zip -d ../input/$dataset

dataset="feedback-prize-effectiveness-st"
kaggle datasets download -d shutotakahashi/$dataset -p ../input
mkdir ../input/$dataset
unzip -o ../input/$dataset.zip -d ../input/$dataset

dataset="fpe-code"
kaggle datasets download -d shutotakahashi/$dataset -p ../input
mkdir ../input/$dataset
unzip -o ../input/$dataset.zip -d ../input/$dataset
```

# Run docker container

## Build image (local)

```sh
cd docker
cp ../src/requirements.txt .
docker build -t kaggle-fpe .
```

## Build image (GCP)

```sh
cd docker
cp ../requirements.txt .
docker build -t kaggle-fpe .
```

## Run (local)

```sh
docker run \
--gpus all \
--rm \
-it \
-v /home/shutotakahashi/projects/kaggle-feedback-prize-effectiveness:/kaggle/working \
-v /home/shutotakahashi/.kaggle:/root/.kaggle \
kaggle-fpe \
bash
```

## Run (GCP)

```sh
docker run \
--gpus all \
--rm \
-it \
-m 85gb \
--shm-size=85gb \
-v /kaggle:/kaggle \
-v /root/.kaggle/:/root/.kaggle/ \
kaggle-fpe \
bash
```