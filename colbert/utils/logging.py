import os
import sys
import ujson
import mlflow
import traceback
import boto3
from torch.utils.tensorboard import SummaryWriter
from colbert.utils.utils import print_message, create_directory


class Logger():
    def __init__(self, rank, run):
        self.rank = rank
        self.is_main = self.rank in [-1, 0]
        self.run = run
        self.logs_path = os.path.join(self.run.path, "logs/")

        if self.is_main:
            self._init_mlflow()
            self.initialized_tensorboard = False
            create_directory(self.logs_path)

    def _init_mlflow(self):
        mlflow.set_tracking_uri('file://' + os.path.join(self.run.experiments_root, "logs/mlruns/"))
        mlflow.set_experiment('/'.join([self.run.experiment, self.run.script]))
        
        mlflow.set_tag('experiment', self.run.experiment)
        mlflow.set_tag('name', self.run.name)
        mlflow.set_tag('path', self.run.path)

    def _init_tensorboard(self):
        root = os.path.join(self.run.experiments_root, "logs/tensorboard/")
        logdir = '__'.join([self.run.experiment, self.run.script, self.run.name])
        logdir = os.path.join(root, logdir)
        self.logdir=logdir

        self.s3_client = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"],  endpoint_url=os.environ["AWS_ENDPOINT_URL"], aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
        self.writer = SummaryWriter(log_dir=logdir)
        self.initialized_tensorboard = True

    def _log_exception(self, etype, value, tb):
        if not self.is_main:
            return

        output_path = os.path.join(self.logs_path, 'exception.txt')
        trace = ''.join(traceback.format_exception(etype, value, tb)) + '\n'
        print_message(trace, '\n\n')

        self.log_new_artifact(output_path, trace)

    def _log_all_artifacts(self):
        if not self.is_main:
            return

        mlflow.log_artifacts(self.logs_path)

    def _log_args(self, args):
        if not self.is_main:
            return

        for key in vars(args):
            value = getattr(args, key)
            if type(value) in [int, float, str, bool]:
                mlflow.log_param(key, value)

        with open(os.path.join(self.logs_path, 'args.json'), 'w') as output_metadata:
            ujson.dump(args.input_arguments.__dict__, output_metadata, indent=4)
            output_metadata.write('\n')

        with open(os.path.join(self.logs_path, 'args.txt'), 'w') as output_metadata:
            output_metadata.write(' '.join(sys.argv) + '\n')

    def log_metric(self, name, value, step, log_to_mlflow=True):
        if not self.is_main:
            return

        if not self.initialized_tensorboard:
            self._init_tensorboard()

        if log_to_mlflow:
            mlflow.log_metric(name, value, step=step)
        self.writer.add_scalar(name, value, step)

        for root,dirs,files in os.walk(self.logdir):
            for file in files:
                self.s3_client.upload_file(os.path.join(root,file), os.environ["ENV AWS_BUCKET"], os.environ["ENV AWS_BUCKET_PATH"]+self.logdir.split('experiments/')[1])
    
    def log_new_artifact(self, path, content):
        with open(path, 'w') as f:
            f.write(content)

        mlflow.log_artifact(path)

    def warn(self, *args):
        msg = print_message('[WARNING]', '\t', *args)

        with open(os.path.join(self.logs_path, 'warnings.txt'), 'a') as output_metadata:
            output_metadata.write(msg + '\n\n\n')

    def info_all(self, *args):
        print_message('[' + str(self.rank) + ']', '\t', *args)

    def info(self, *args):
        if self.is_main:
            print_message(*args)
