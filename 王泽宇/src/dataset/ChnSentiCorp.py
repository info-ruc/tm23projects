import csv

import datasets


_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1uV-aDQoMI51A27OxVgJnzxqZFQqkDydZ&export=download"
_DEV_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1kI_LUYm9m0pVGDb4LHqtjwauUrIGo_rC&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1YJkSqGFeo-8ifmgVbxGMTte7Y-yxPHg7&export=download"



class ChnSentiCorp(datasets.GeneratorBasedBuilder):


    def _info(self):
        return datasets.DatasetInfo(
            description=None,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["negative", "positive"])
                }
            ),
            homepage=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        dev_path = dl_manager.download_and_extract(_DEV_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    continue
                label, text = row
                label = int(label)
                yield id_, {"text": text, "label": label}