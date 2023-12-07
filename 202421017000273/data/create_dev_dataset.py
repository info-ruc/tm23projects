# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import random

prob_selection = 0.1

total_samples = []
with open('atis.train+dev.txt', 'r') as fr:
	total_samples = fr.readlines()

train_samples = []
dev_samples = []
for line in total_samples:
	if random.random() < prob_selection:
		dev_samples.append(line)
	else:
		train_samples.append(line)

with open('atis.train.txt', 'w') as fr:
	for line in train_samples:
		fr.write(line)

with open('atis.dev.txt', 'w') as fr:
	for line in dev_samples:
		fr.write(line)