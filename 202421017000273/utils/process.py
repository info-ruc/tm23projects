# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
from torch.nn import NLLLoss
from torch.optim import SGD, Adam, Adagrad
from torch.autograd import Variable

import time

'''
给定模型, 数据集, 优化工具, 损失计算对象和其它参数
训练模型.
'''


def train(encoder, decoder, dataset, optim_mode,
		  batch_size, learning_rate,
		  train_epoch, print_each, save_each, validate_each,
		  model_save_dir):
	total_time_start = time.time()

	'''
	检查环境是否支持 GPU 加速计算.
	'''
	if torch.cuda.is_available():
		time_start = time.time()
		encoder = encoder.cuda()
		decoder = decoder.cuda()

		print("模型 Encoder, Decoder 已加入 GPU, 共用时 {:.6f} 秒.\n\n".format(time.time() - time_start))

	criterion = NLLLoss()
	if optim_mode == "adam":
		en_optimer = Adam(encoder.parameters(), lr=learning_rate)
		de_optimer = Adam(decoder.parameters(), lr=learning_rate)
	elif optim_mode == "sgd":
		en_optimer = SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
		de_optimer = SGD(decoder.parameters(), lr=learning_rate, momentum=0.9)
	elif optim_mode == "adagrad":
		en_optimer = Adagrad(encoder.parameters(), lr=learning_rate)
		de_optimer = Adagrad(decoder.parameters(), lr=learning_rate)
	else:
		raise Exception("优化方法只能选取 adam, sgd 和 adagrad, 注意大小写.")

	for epoch in range(0, train_epoch + 1):
		total_loss = Variable(torch.FloatTensor([0]))
		sentence_batch, labels_batch, seq_lens, intent_batch = dataset.get_batch(batch_size)

		sent_var = Variable(torch.LongTensor(sentence_batch))
		labels_var_list = []
		for labels in labels_batch:
			labels_var_list.append(Variable(torch.LongTensor(labels)))
		intent_var = Variable(torch.LongTensor(intent_batch))

		if torch.cuda.is_available():
			time_start = time.time()
			sent_var = sent_var.cuda()
			labels_var_list = [elem.cuda() for elem in labels_var_list]
			intent_var = intent_var.cuda()
			time_con = time.time() - time_start

			print('batch {} 已载入 GPU, 用时 {:.6f} 秒, 正在前向计算和反馈更新.'.format(epoch, time_con))
		else:
			print('batch {} 已载入 CPU, 正在前向计算和反馈更新.'.format(epoch))

		time_start = time.time()
		hiddens, last_hidden = encoder(sent_var, seq_lens)
		slot_pred_list, intent_pred = decoder(last_hidden, hiddens, seq_lens)
		time_con = time.time() - time_start
		print('batch {} 的前向计算时间开销为 {:.6f} 秒.'.format(epoch, time_con))

		time_start = time.time()
		#  NLLLoss 损失.
		total_loss = criterion(intent_pred, intent_var)
		for (slot_pred, label_var) in zip(slot_pred_list, labels_var_list):
			total_loss += criterion(slot_pred, label_var)

		# 使用平均损失.
		total_loss /= batch_size

		# 梯度清零, 反向求梯度, 参数更新.
		en_optimer.zero_grad()
		de_optimer.zero_grad()
		total_loss.backward()
		en_optimer.step()
		de_optimer.step()
		time_con = time.time() - time_start
		print('batch {} 的回馈更新时间开销为 {:.6f} 秒.\n'.format(epoch, time_con))

		# 打印当前 batch 的损失.
		if epoch % print_each == 0:
			print('在第 {} 轮 batch 的平均损失为 {:.6f}.\n'.format(epoch, total_loss.cpu().data.numpy().item()))

			if epoch % validate_each != 0:
				time.sleep(5)

		if epoch % save_each == 0:
			torch.save(encoder, model_save_dir + 'encoder.pth')
			torch.save(decoder, model_save_dir + 'decoder.pth')

		if epoch % validate_each == 0:
			time_start = time.time()

			slot_accuracy, slot_f1, intent_accuracy = test(encoder, decoder, dataset)
			print('在第 {} 轮验证数据集中 slot filling 的 准确率为 {:.6f}, F1 值为 {:6f}.'.format(epoch, slot_accuracy,
																								  slot_f1))
			print('在第 {} 轮验证数据集中 intent detection 的 准确率为 {:.6f}.'.format(epoch, intent_accuracy))

			time_con = time.time() - time_start
			print('在 {} 轮 batch 中, 用于在测试集上评测的时间为 {:.6f} 秒.'.format(epoch, time_con))

			time.sleep(5)

		print('\n')

	total_time_con = time.time() - total_time_start
	print('本次训练 + 测试共计用时 {:.6f} 秒.'.format(total_time_con))

	return devset_evaluation(encoder, decoder, dataset)


def test(encoder, decoder, dataset):
	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()

	sentence_list, labels_list, seq_lens, intent_list = dataset.get_test()

	sent_var = Variable(torch.LongTensor(sentence_list))
	if torch.cuda.is_available():
		sent_var = sent_var.cuda()

	# encoder 和 decoder 的前向计算过程.
	all_hiddens, last_hidden = encoder(sent_var, seq_lens)
	slot_pred_list, intent_pred = decoder(last_hidden, all_hiddens, seq_lens)

	# 得到 slot 的预测.
	slot_prediction = []
	for slot_pred in slot_pred_list:
		_, idxs = slot_pred.topk(1, dim=1)
		slot_prediction.append(idxs.cpu().data.transpose(0, 1).numpy().tolist()[0])

	# 得到 intent 的预测.
	_, idxs = intent_pred.topk(1, dim=1)
	intent_prediction = idxs.cpu().data.transpose(0, 1).numpy().tolist()[0]

	# 计算 slot 的 acc 和 f1.
	slot_acc = accuracy(ravel_list(slot_prediction), ravel_list(labels_list))
	slot_f1 = slot_f1_measure(slot_prediction, labels_list, dataset.get_alphabets()[1])

	# 计算 intent 的 acc 和 f1.
	intent_acc = accuracy(intent_prediction, intent_list)

	return slot_acc, slot_f1, intent_acc


def devset_evaluation(encoder, decoder, dataset):
	time_start = time.time()

	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()

	sentence_list, labels_list, seq_lens, intent_list = dataset.get_dev()

	sent_var = Variable(torch.LongTensor(sentence_list))
	if torch.cuda.is_available():
		sent_var = sent_var.cuda()

	# encoder 和 decoder 的前向计算过程.
	all_hiddens, last_hidden = encoder(sent_var, seq_lens)
	slot_pred_list, intent_pred = decoder(last_hidden, all_hiddens, seq_lens)

	# 得到 slot 的预测.
	slot_prediction = []
	for slot_pred in slot_pred_list:
		_, idxs = slot_pred.topk(1, dim=1)
		slot_prediction.append(idxs.cpu().data.transpose(0, 1).numpy().tolist()[0])

	# 得到 intent 的预测.
	_, idxs = intent_pred.topk(1, dim=1)
	intent_prediction = idxs.cpu().data.transpose(0, 1).numpy().tolist()[0]

	# 计算 slot 的 acc 和 f1.
	slot_acc = accuracy(ravel_list(slot_prediction), ravel_list(labels_list))
	slot_f1 = slot_f1_measure(slot_prediction, labels_list, dataset.get_alphabets()[1])

	# 计算 intent 的 acc 和 f1.
	intent_acc = accuracy(intent_prediction, intent_list)

	time_con = time.time() - time_start

	print('\n开发集(dev)的时间总开销(加载, 前向计算等)为 {:.6f} 秒.'.format(time_con))
	print('其中在 slot filling 的任务上, 准确率为 {:.6f}, F1 得分为 {:.6f},'.format(slot_acc, slot_f1))
	print('在 intent detection 的任务上, 准确率为 {:.6f}.'.format(intent_acc))

	return slot_acc, slot_f1, intent_acc


def accuracy(pred_list, real_list):
	count = 0
	for pred, real in zip(pred_list, real_list):
		if pred == real:
			count += 1

	return 1.0 * count / len(pred_list)


def slot_f1_measure(pred_list, label_list, alphabet):
	# 计算 True Postive, False Positive 和 False Negative 的值.
	tp = 0.0
	fp = 0.0
	fn = 0.0
	for i in range(len(pred_list)):
		seg = set()
		result = [alphabet.words(tag) for tag in pred_list[i]]
		target = [alphabet.words(tag) for tag in label_list[i]]

		j = 0
		while j < len(target):
			cur = target[j]
			if cur[0] == 'B':
				k = j + 1
				while k < len(target):
					str_ = target[k]
					if not (str_[0] == 'I' and cur[1:] == str_[1:]):
						break
					k = k + 1
				seg.add((cur, j, k - 1))
				j = k - 1
			j = j + 1

		tp_ = 0
		j = 0
		while j < len(result):
			cur = result[j]
			if cur[0] == 'B':
				k = j + 1
				while k < len(result):
					str_ = result[k]
					if not (str_[0] == 'I' and cur[1:] == str_[1:]):
						break
					k = k + 1
				if (cur, j, k - 1) in seg:
					tp_ += 1
				else:
					fp += 1
				j = k - 1
			j = j + 1

		fn += len(seg) - tp_
		tp += tp_

	P = tp / (tp + fp) if tp + fp != 0 else 0
	R = tp / (tp + fn)
	F = 2 * P * R / (P + R) if P + R != 0 else 0

	return F


'''
对测试集的样例做预测. 或者按照格式
sentence_list, labels_list, seq_lens, intent_list
给出一个 tuple. sentence_list 中的样本随着 index 增大
其长度不增. give_predictions >= 2, 否则报错.
'''


def predict(dataset, encoder=None, decoder=None, sample_tuple=None,
			name=None, give_predictions=5):
	time_start = time.time()

	if encoder is None or decoder is None:
		encoder = torch.load('./save/model/encoder_.pth')
		decoder = torch.load('./save/model/decoder_.pth')

	if sample_tuple is None:
		sentence_list, labels_list, seq_lens, intent_list = dataset.get_test()
	else:
		sentence_list, labels_list, seq_lens, intent_list = sample_tuple

	sent_var = Variable()
	if torch.cuda.is_available():
		sent_var = sent_var.cuda
		encoder = encoder.cuda()
		decoder = decoder.cuda()

	all_hiddens, last_hidden = encoder(sent_var, seq_lens)
	slot_pred_list, intent_pred = decoder(last_hidden, all_hiddens, seq_lens)

	slot_prediction = []
	for slot_pred in slot_pred_list:
		_, idxs = slot_pred.topk(give_predictions, dim=1)
		slot_prediction.append(idxs.cpu().data.numpy().tolist())

	_, idxs = intent_pred.topk(give_predictions, dim=1)
	intent_prediction = idxs.cpu().data.numpy().tolist()

	if name is None:
		file_path = './save/prediction/test.txt'
	else:
		file_path = './save/prediction/' + name + '.txt'

	word_dict, label_dict, intent_dict = dataset.get_alphabets()

	with open(file_path, 'a') as fr:
		for idx in range(0, len(sentence_list)):
			sentence = word_dict.words(sentence_list[idx])
			real_slots = label_dict.words(labels_list[idx])
			real_intent = intent_dict.words(intent_list[idx])

			predict_slots = label_dict.words(slot_prediction[idx])
			predict_intents = intent_dict.words(intent_prediction[idx])

			for jdx in range(0, seq_lens[idx]):
				# print(sentence[jdx], real_slots[jdx])
				fr.write(sentence[jdx] + '\t' + real_slots[jdx] + ' ')
				for slot in predict_slots[jdx]:
					fr.write(slot + ' ')

				fr.write('\n')

			fr.write(real_intent + ' ')
			for intent in predict_intents:
				fr.write(intent + ' ')

			fr.write('\n\n')

	time_con = time.time() - time_start
	print('预测文件放在路径 {} 下, 共消耗时间 {:.6f} 秒.'.format(file_path, time_con))


def ravel_list(l):
	new_l = []
	for elem in l:
		new_l.extend(elem)
	return new_l
