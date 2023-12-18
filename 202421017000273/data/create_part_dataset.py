# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import random

p = 0.04

with open('atis.all.txt', 'r') as fr:
	with open('atis.part.txt', 'w') as fw:
		for line in fr.readlines():
			if random.random() < p:
				fw.write(line)