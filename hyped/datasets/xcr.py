import os
import re
import json
import datasets
import urllib3
from dataclasses import dataclass

@dataclass
class XCRepoConfig(datasets.BuilderConfig):
    # google drive file id
    drive_id:str = None
    # file structure
    trn_file:str = None
    tst_file:str = None
    lbl_file:str = None
    # split sizes
    trn_size:int = None
    tst_size:int = None

class XCRepoDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = XCRepoConfig

    BUILDER_CONFIGS = [
        XCRepoConfig(
            name="AmazonCat-13K",
            drive_id="17rVRDarPwlMpb3l5zof9h34FlwbpTu4l",
            trn_file="AmazonCat-13K.raw/trn.json.gz",
            tst_file="AmazonCat-13K.raw/tst.json.gz",
            lbl_file="AmazonCat-13K.raw/Yf.txt",
            trn_size=1186239,
            tst_size=306782
        ),
        XCRepoConfig(
            name="AmazonCat-14K",
            drive_id="1vy1N-lDdDfuoo0CNwFE11hb3INCpJHFx",
            trn_file="AmazonCat-14K.raw/trn.json.gz",
            tst_file="AmazonCat-14K.raw/tst.json.gz",
            lbl_file="AmazonCat-14K.raw/Yf.txt",
            trn_size=4398050,
            tst_size=1099725
        )
    ]

    def _info(self):

        # check if config is valid
        if self.config.drive_id is None:
            raise ValueError("No config specified")

        return datasets.DatasetInfo(
            builder_name=self.config.name,
            description="",
            features=datasets.Features({
                'title': datasets.Value('string'),
                'text': datasets.Value('string'),
                'labels': datasets.Sequence(datasets.Value('string'))
            }),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):

        # dataset_id = "0B1HXnM1lBuoqMzVhZjcwNTAtZWI5OS00ZDg3LWEyMzktNzZmYWY2Y2NhNWQx"
        dataset_url = "https://docs.google.com/uc?export=download&id=%s" % self.config.drive_id

        # do a header request to check whether the url actually downloads
        # the content or redirects to the file to large warning page
        resp = urllib3.request('HEAD', dataset_url)

        # check by content type
        if resp.headers['Content-Type'].startswith('text/html'):
            # html response means the response is the warning page
            # so lets download the html code
            with open(dl_manager.download(dataset_url), 'r') as f:
                page = f.read()
            # extract the confirmation code from the warning page
            pattern = r"(?<=confirm=)t&amp;uuid=\w{8}-\w{4}-\w{4}-\w{4}-\w{12}"
            match = re.search(pattern, page)
            # check if confirmation code is found
            if match is None:
                raise RuntimeError("Confirmation code not found!")
            # add confirmation code to url
            dataset_url = "https://docs.google.com/uc?export=download&id=%s&confirm=%s" % (self.config.drive_id, match.group())

        # download dataset archive and extract
        dataset_path = dl_manager.download_and_extract(dataset_url)

        # build full file paths
        trn_fpath = os.path.join(dataset_path, self.config.trn_file)
        tst_fpath = os.path.join(dataset_path, self.config.tst_file)
        lbl_fpath = os.path.join(dataset_path, self.config.lbl_file)
        # unpack all files
        trn_fpath = dl_manager.extract(trn_fpath)
        tst_fpath = dl_manager.extract(tst_fpath)

        # load dataset names
        with open(lbl_fpath, 'r', encoding='latin-1') as f:
            labels = f.read().strip().splitlines()
        # this is really hacky but seems to be the only way
        # to update the dataset features after downloading
        self.info.features['labels'] = datasets.Sequence(datasets.ClassLabel(names=labels))
        # train and test split generator
        trn_split = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs=dict(fpath=trn_fpath)
        )
        tst_split = datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs=dict(fpath=tst_fpath)
        )
        # set split sizes for tqdm
        trn_split.split_info.num_examples = self.config.trn_size
        tst_split.split_info.num_examples = self.config.tst_size
        # return splits
        return [trn_split, tst_split]

    def _generate_examples(self, fpath):
        # open data file
        with open(fpath, 'r', encoding='latin-1') as f:
            # json-decode each line
            for i, example in enumerate(map(json.loads, f)):
                # extract information from example
                yield i, {
                    'title': example['title'],
                    'text': example['content'],
                    'labels': example['target_ind']
                }
