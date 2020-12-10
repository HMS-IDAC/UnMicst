import numpy as np
from scipy import misc
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
import shutil
import scipy.io as sio
import os, fnmatch, glob
import skimage.exposure as sk
import skimage.io
import argparse
import czifile
from nd2reader import ND2Reader
import tifffile
import sys, math
tf.disable_v2_behavior()
# sys.path.insert(0, 'C:\\Users\\Public\\Documents\\ImageScience')
from toolbox.imtools import *
from toolbox.ftools import *
from toolbox.PartitionOfImage import PI2D
from toolbox import GPUselect

def concat3(lst):
	return tf.concat(lst, 3)


class UNet2D:
	hp = None  # hyper-parameters
	nn = None  # network
	tfTraining = None  # if training or not (to handle batch norm)
	tfData = None  # data placeholder
	Session = None
	DatasetMean = 0
	DatasetStDev = 0

	def setupWithHP(hp):
		UNet2D.setup(hp['imSize'],
					 hp['nChannels'],
					 hp['nClasses'],
					 hp['nOut0'],
					 hp['featMapsFact'],
					 hp['downSampFact'],
					 hp['ks'],
					 hp['nExtraConvs'],
					 hp['stdDev0'],
					 hp['nLayers'],
					 hp['batchSize'])

	def setup(imSize, nChannels, nClasses, nOut0, featMapsFact, downSampFact, kernelSize, nExtraConvs, stdDev0,
			  nDownSampLayers, batchSize):
		UNet2D.hp = {'imSize': imSize,
					 'nClasses': nClasses,
					 'nChannels': nChannels,
					 'nExtraConvs': nExtraConvs,
					 'nLayers': nDownSampLayers,
					 'featMapsFact': featMapsFact,
					 'downSampFact': downSampFact,
					 'ks': kernelSize,
					 'nOut0': nOut0,
					 'stdDev0': stdDev0,
					 'batchSize': batchSize}

		nOutX = [UNet2D.hp['nChannels'], UNet2D.hp['nOut0']]
		dsfX = []
		for i in range(UNet2D.hp['nLayers']):
			nOutX.append(nOutX[-1] * UNet2D.hp['featMapsFact'])
			dsfX.append(UNet2D.hp['downSampFact'])

		# --------------------------------------------------
		# downsampling layer
		# --------------------------------------------------
		with tf.name_scope('placeholders'):
			UNet2D.tfTraining = tf.placeholder(tf.bool, name='training')
			UNet2D.tfData = tf.placeholder("float", shape=[None, UNet2D.hp['imSize'], UNet2D.hp['imSize'],
														   UNet2D.hp['nChannels']], name='data')

		def down_samp_layer(data, index):
			regularizer = tf.keras.regularizers.l1(0.00008)
			with tf.variable_scope('ld%d' % index):
				ldXWeights1 = tf.Variable(
					tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index], nOutX[index + 1]],
										stddev=stdDev0), name='kernelD%d' % index)
				# ldXWeights1 = tf.get_variable(
				# 	initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
				# 	shape=[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index], nOutX[index + 1]], name='kernelD%d' % index, regularizer=regularizer)
				ldXWeightsExtra = []
				for i in range(nExtraConvs):
					ldXWeightsExtra.append(
						tf.get_variable(initializer=tf.compat.v1.keras.initializers.VarianceScaling(mode='fan_in'),
										shape=[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index + 1], nOutX[index + 1]],
										name='kernelExtra%d' % i))
					# ldXWeightsExtra.append(tf.Variable(
					# 	tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index + 1], nOutX[index + 1]],
					# 						stddev=stdDev0), name='kernelExtra%d' % i))

				c00 = tf.nn.conv2d(data, ldXWeights1, strides=[1, 1, 1, 1], padding='SAME')
				for i in range(nExtraConvs):
					c00 = tf.nn.conv2d(tf.nn.leaky_relu(c00), ldXWeightsExtra[i], strides=[1, 1, 1, 1], padding='SAME')

				ldXWeightsShortcut = tf.get_variable(initializer=tf.compat.v1.keras.initializers.VarianceScaling(mode='fan_in'),
													 shape=[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index],
															nOutX[index + 1]],
													 name='shortcutWeights', regularizer=regularizer)
				# ldXWeightsShortcut = tf.Variable(
				# 	tf.truncated_normal([1, 1, nOutX[index], nOutX[index + 1]], stddev=stdDev0), name='shortcutWeights')
				shortcut = tf.nn.conv2d(data, ldXWeightsShortcut, strides=[1, 1, 1, 1], padding='SAME')

				bn = tf.nn.leaky_relu(tf.layers.batch_normalization(c00+shortcut, training=UNet2D.tfTraining))
				# bn = tf.layers.batch_normalization(tf.nn.leaky_relu(c00 + shortcut), training=UNet2D.tfTraining)
				# bn = tf.layers.dropout(bn, 0.05 * index, training=UNet2D.tfTraining)
				return tf.nn.max_pool(bn, ksize=[1, dsfX[index], dsfX[index], 1],
									  strides=[1, dsfX[index], dsfX[index], 1], padding='SAME', name='maxpool')

		# --------------------------------------------------
		# bottom layer
		# --------------------------------------------------

		with tf.variable_scope('lb'):
			regularizer = tf.keras.regularizers.l1(0.00008)
			lbWeights1 = tf.get_variable(initializer=tf.compat.v1.keras.initializers.VarianceScaling(mode='fan_in'),
										 shape=[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[UNet2D.hp['nLayers']],
												nOutX[UNet2D.hp['nLayers'] + 1]],
										 name='kernel1', regularizer=regularizer)
			# lbWeights1 = tf.Variable(tf.truncated_normal(
			# 	[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[UNet2D.hp['nLayers']], nOutX[UNet2D.hp['nLayers'] + 1]],
			# 	stddev=stdDev0), name='kernel1')
			def lb(hidden):
				# lbn= tf.nn.leaky_relu(
				# 	tf.nn.conv2d(hidden, lbWeights1, strides=[1, 1, 1, 1], padding='SAME'), name='conv')
				lbn = tf.nn.leaky_relu(tf.layers.batch_normalization(
					tf.nn.conv2d(hidden, lbWeights1, strides=[1, 1, 1, 1], padding='SAME'),
					name='conv',training=UNet2D.tfTraining))
				return tf.layers.dropout(lbn, 0.35, training=UNet2D.tfTraining)

		# --------------------------------------------------
		# downsampling
		# --------------------------------------------------

		with tf.name_scope('downsampling'):
			dsX = []
			dsX.append(UNet2D.tfData)

			for i in range(UNet2D.hp['nLayers']):
				dsX.append(down_samp_layer(dsX[i], i))

			b = lb(dsX[UNet2D.hp['nLayers']])

		# --------------------------------------------------
		# upsampling layer
		# --------------------------------------------------

		def up_samp_layer(data, index):
			with tf.variable_scope('lu%d' % index):
				regularizer = tf.keras.regularizers.l1(0.00008)
				luXWeights1 = tf.get_variable(
					initializer=tf.compat.v1.keras.initializers.VarianceScaling(mode='fan_in'),
					shape=[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index + 1], nOutX[index + 2]],
					name='kernelU%d' % index,regularizer=regularizer)
				luXWeights2 = tf.get_variable(
					initializer=tf.compat.v1.keras.initializers.VarianceScaling(mode='fan_in'),
					shape=[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index] + nOutX[index + 1], nOutX[index + 1]],
					name='kernel2',regularizer=regularizer)
				luXWeightsExtra = []
				for i in range(nExtraConvs):
					luXWeightsExtra.append(tf.get_variable(
						initializer=tf.compat.v1.keras.initializers.VarianceScaling(mode='fan_in'),
						shape=[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index + 1], nOutX[index + 1]],
						name='kernel2Extra%d' % i))
				# luXWeights1 = tf.Variable(
				# 	tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index + 1], nOutX[index + 2]],
				# 						stddev=stdDev0), name='kernel1')
				# luXWeights2 = tf.Variable(tf.truncated_normal(
				# 	[UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index] + nOutX[index + 1], nOutX[index + 1]],
				# 	stddev=stdDev0), name='kernel2')
				# luXWeightsExtra = []
				# for i in range(nExtraConvs):
				# 	luXWeightsExtra.append(tf.Variable(
				# 		tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index + 1], nOutX[index + 1]],
				# 							stddev=stdDev0), name='kernel2Extra%d' % i))

				outSize = UNet2D.hp['imSize']
				for i in range(index):
					outSize /= dsfX[i]
				outSize = int(outSize)

				outputShape = [UNet2D.hp['batchSize'], outSize, outSize, nOutX[index + 1]]
				us = tf.nn.leaky_relu(
					tf.nn.conv2d_transpose(data, luXWeights1, outputShape, strides=[1, dsfX[index], dsfX[index], 1],
										   padding='SAME'), name='conv1')
				cc = concat3([dsX[index], us])
				# cv = tf.nn.leaky_relu(tf.nn.conv2d(cc, luXWeights2, strides=[1, 1, 1, 1], padding='SAME'), name='conv2')
				cv = tf.nn.leaky_relu(
					tf.layers.batch_normalization(tf.nn.conv2d(cc, luXWeights2, strides=[1, 1, 1, 1], padding='SAME'),
												  name='conv2',
												  training=UNet2D.tfTraining))
				for i in range(nExtraConvs):
					cv = tf.nn.leaky_relu(tf.nn.conv2d(cv, luXWeightsExtra[i], strides=[1, 1, 1, 1], padding='SAME'),
										  name='conv2Extra%d' % i)
				# cv = tf.layers.dropout(cv, 0.25 - 0.05 * index, training=UNet2D.tfTraining)
				return cv

		# --------------------------------------------------
		# final (top) layer
		# --------------------------------------------------

		with tf.variable_scope('lt'):
			regularizer = tf.keras.regularizers.l1(0.00008)
			# ltWeights1 = tf.Variable(tf.truncated_normal([1, 1, nOutX[1], nClasses], stddev=stdDev0), name='kernel')
			ltWeights1 = tf.get_variable(initializer=tf.compat.v1.keras.initializers.VarianceScaling(mode='fan_in'),
										 shape=[1, 1, nOutX[1], nClasses],
										 name='kernel',regularizer=regularizer)
			def lt(hidden):
				# return tf.nn.conv2d(hidden, ltWeights1, strides=[1, 1, 1, 1], padding='SAME', name='conv')
				return tf.layers.batch_normalization(
					tf.nn.conv2d(hidden, ltWeights1, strides=[1, 1, 1, 1], padding='SAME', name='conv'),
					training=UNet2D.tfTraining)
		# --------------------------------------------------
		# upsampling
		# --------------------------------------------------

		with tf.name_scope('upsampling'):
			usX = []
			usX.append(b)

			for i in range(UNet2D.hp['nLayers']):
				usX.append(up_samp_layer(usX[i], UNet2D.hp['nLayers'] - 1 - i))

			t = lt(usX[UNet2D.hp['nLayers']])

		sm = tf.nn.softmax(t, -1)
		UNet2D.nn = sm


	def train(imPath, validPath, testPath, logPath, modelPath, pmPath, nTrain, nValid, nTest, restoreVariables, nSteps, gpuIndex,
			  testPMIndex):
		os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuIndex

		outLogPath = logPath
		trainWriterPath = pathjoin(logPath, 'Train')
		validWriterPath = pathjoin(logPath, 'Valid')
		outModelPath = pathjoin(modelPath, 'model.ckpt')
		outPMPath = pmPath

		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']
		nClasses = UNet2D.hp['nClasses']

		# --------------------------------------------------
		# data
		# --------------------------------------------------
		nAug =12
		Train = np.zeros((nTrain, imSize, imSize, nAug,nChannels))
		Valid = np.zeros((nValid, imSize, imSize,nAug,nChannels))
		Test = np.zeros((nTest, imSize, imSize, nAug,nChannels))
		LTrain = np.zeros((nTrain, imSize, imSize, nClasses))
		LValid = np.zeros((nValid, imSize, imSize, nClasses))
		LTest = np.zeros((nTest, imSize, imSize, nClasses))
		WTrain = np.zeros((nTrain, imSize, imSize, nClasses))
		WValid = np.zeros((nValid, imSize, imSize, nClasses))
		WTest = np.zeros((nTest, imSize, imSize, nClasses))

		print('loading data, computing mean / st dev')
		if not os.path.exists(modelPath):
			os.makedirs(modelPath)
		# if restoreVariables:
		# 	datasetMean = loadData(pathjoin(modelPath,'datasetMean.data'))
		# 	datasetStDev = loadData(pathjoin(modelPath,'datasetStDev.data'))
		# else:
		datasetMean = 0.34
		datasetStDev = 0.25
		bgWeight = 1
		contourWeight = 2
		nucleiWeight = 7#5
		intersectWeight = 15

		# for iSample in range(nTrain + nValid + nTest):
		#     I = im2double(tifread('%s/I%05d_Img.tif' % (imPath, iSample)))
		#     datasetMean += np.mean(I)
		#     datasetStDev += np.std(I)
		# datasetMean /= (nTrain + nValid + nTest)
		# datasetStDev /= (nTrain + nValid + nTest)
		saveData(datasetMean, pathjoin(modelPath, 'datasetMean.data'))
		saveData(datasetStDev, pathjoin(modelPath, 'datasetStDev.data'))

		perm = np.arange(nTrain)
		np.random.shuffle(perm)

		for iSample in range(0, nTrain):
			path = '%s/I%05d_Img.tif' % (imPath, perm[iSample])
			for iChan in range(nChannels):
				for iAug in range(nAug):
					im = im2double(skio.imread(path, img_num=iAug + nAug*iChan))
					Train[iSample, :, :, iAug, iChan] = (im - datasetMean) / datasetStDev
			path = '%s/I%05d_Ant.tif' % (imPath, perm[iSample])
			im = tifread(path)
			path = '%s/I%05d_wt.tif' % (imPath, perm[iSample])
			W = tifread(path)
			for i in range(nClasses):
				LTrain[iSample, :, :, i] = (im == i + 1)
				if i == 1:
					WTrain[iSample, :, :, i] = (W * intersectWeight) + contourWeight
				elif i == 2:
					WTrain[iSample, :, :, i] = (W * 0) + nucleiWeight
				else:
					WTrain[iSample, :, :, i] = (W * 0) + bgWeight

		permV = np.arange(nValid)
		np.random.shuffle(permV)
		for iSample in range(0, nValid):
			path = '%s/I%05d_Img.tif' % (validPath, permV[iSample])
			# im = im2double(tifread(path))
			for iChan in range(nChannels):
				for iAug in range(nAug):
					im = im2double(skio.imread(path, img_num=iAug + nAug*iChan))
					Valid[iSample, :, :, iAug, iChan] = (im - datasetMean) / datasetStDev
			path = '%s/I%05d_Ant.tif' % (validPath, permV[iSample])
			im = tifread(path)
			path = '%s/I%05d_wt.tif' % (validPath,  permV[iSample])
			W = tifread(path)
			for i in range(nClasses):
				LValid[iSample, :, :, i] = (im == i + 1)
				if i == 1:
					WValid[iSample, :, :, i] = (W * intersectWeight) + contourWeight
				elif i == 2:
					WValid[iSample, :, :, i] = (W * 0) + nucleiWeight
				else:
					WValid[iSample, :, :, i] = (W * 0) + bgWeight

		for iSample in range(0, nTest):
			path = '%s/I%05d_Img.tif' % (testPath, iSample)
			for iChan in range(nChannels):
				for iAug in range(nAug):
					im = im2double(skio.imread(path, img_num=iAug + nAug*iChan))
					Test[iSample, :, :, iAug, iChan] = (im - datasetMean) / datasetStDev
			path = '%s/I%05d_Ant.tif' % (testPath, iSample)
			im = tifread(path)
			path = '%s/I%05d_wt.tif' % (testPath, iSample)
			W = tifread(path)
			for i in range(nClasses):
				LTest[iSample, :, :, i] = (im == i + 1)
				if i == 1:
					WTest[iSample, :, :, i] = (W * intersectWeight) + contourWeight
				elif i == 2:
					WTest[iSample, :, :, i] = (W * 0) + nucleiWeight
				else:
					WTest[iSample, :, :, i] = (W * 0) + bgWeight

		# --------------------------------------------------
		# optimization
		# --------------------------------------------------

		tfLabels = tf.placeholder("float", shape=[None, imSize, imSize, nClasses], name='labels')
		tfWeights = tf.placeholder("float", shape=[None, imSize, imSize, nClasses], name='weights')
		globalStep = tf.Variable(0, trainable=False)
		learningRate0 = 0.00005
		decaySteps = 5000
		decayRate = 0.98
		learningRate = tf.train.exponential_decay(learningRate0, globalStep, decaySteps, decayRate, staircase=True)

		with tf.name_scope('optim'):
			l2_loss = tf.losses.get_regularization_loss()
			eps = 1e-7
			log_p = tf.log(tf.clip_by_value(UNet2D.nn, eps, 1 - eps))
			loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tf.cast(tfWeights, tf.float32),
															 tf.multiply(tf.cast(tfLabels, tf.float32),
																		 log_p)), 3)) + l2_loss

			updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			# optimizer = tf.train.MomentumOptimizer(1e-3,0.9)
			# optimizer = tf.train.MomentumOptimizer(learningRate,0.99)
			optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
			with tf.control_dependencies(updateOps):
				optOp = optimizer.minimize(loss, global_step=globalStep)

		# for g, v in gradients:
		# 	tf.summary.histogram(v.name, v)
		# 	tf.summary.histogram(v.name + '_grad', g)

		with tf.name_scope('eval'):
			error = []
			for iClass in range(nClasses):
				labels0 = tf.reshape(tf.to_int32(tf.slice(tfLabels, [0, 0, 0, iClass], [-1, -1, -1, 1])),
									 [batchSize, imSize, imSize])
				predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(UNet2D.nn, 3), iClass)),
									  [batchSize, imSize, imSize])
				correct = tf.multiply(labels0, predict0)
				nCorrect0 = tf.reduce_sum(correct)
				nLabels0 = tf.reduce_sum(labels0)
				error.append(1 - tf.to_float(nCorrect0) / tf.to_float(nLabels0))
			errors = tf.tuple(error)

		# --------------------------------------------------
		# inspection
		# --------------------------------------------------

		with tf.name_scope('scalars'):
			tf.summary.scalar('avg_cross_entropy', loss)
			for iClass in range(nClasses):
				tf.summary.scalar('avg_pixel_error_%d' % iClass, error[iClass])
			tf.summary.scalar('learning_rate', learningRate)
		with tf.name_scope('images'):
			split0 = tf.slice(UNet2D.nn, [0, 0, 0, 1], [-1, -1, -1, 1])
			split1 = tf.slice(UNet2D.tfData, [0, 0, 0, 0], [-1, -1, -1, 1])

			planeImN = tf.div(tf.subtract(split1, tf.reduce_min(split1, axis=(1, 2), keep_dims=True)),
							  tf.subtract(tf.reduce_max(split1, axis=(1, 2), keep_dims=True),
										  tf.reduce_min(split1, axis=(1, 2), keep_dims=True)))
			# splitL = tf.slice(UNet2D.tfData, [0, 0, 0, 1], [-1, -1, -1, 1])
			# planeImN2 = tf.div(tf.subtract(splitL, tf.reduce_min(splitL, axis=(1, 2), keep_dims=True)),
			# 				  tf.subtract(tf.reduce_max(splitL, axis=(1, 2), keep_dims=True),
			# 							  tf.reduce_min(splitL, axis=(1, 2), keep_dims=True)))

			plane = tf.concat([planeImN, split0], 2)
			split2 = tf.slice(UNet2D.nn, [0, 0, 0, 2], [-1, -1, -1, 1])

			# planeImN2 = tf.div(tf.subtract(split3, tf.reduce_min(split3, axis=(1, 2), keep_dims=True)),
			#                   tf.subtract(tf.reduce_max(split3, axis=(1, 2), keep_dims=True),
			#                               tf.reduce_min(split3, axis=(1, 2), keep_dims=True)))
			plane = tf.concat([plane, split2], 2)
			tf.summary.image('impm', plane, max_outputs=4)
		merged = tf.summary.merge_all()

		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		sess = tf.Session(config=config)  # config parameter needed to save variables when using GPU

		if os.path.exists(outLogPath):
			shutil.rmtree(outLogPath)
		trainWriter = tf.summary.FileWriter(trainWriterPath, sess.graph)
		validWriter = tf.summary.FileWriter(validWriterPath, sess.graph)

		if restoreVariables:
			saver.restore(sess, outModelPath)
			print("Model restored.")
		else:
			sess.run(tf.global_variables_initializer())

		# --------------------------------------------------
		# train
		# --------------------------------------------------

		batchData = np.zeros((batchSize, imSize, imSize, nChannels))
		batchLabels = np.zeros((batchSize, imSize, imSize, nClasses))
		batchWeights = np.zeros((batchSize, imSize, imSize, nClasses))
		permT = np.arange(nTrain)
		np.random.shuffle(permT)

		permV = np.arange(nValid)
		np.random.shuffle(permV)

		maxBrig = 1 * datasetStDev
		maxCont = 0.1 * datasetStDev
		jT = 0
		jV = 0
		epochCounter = 1
		for i in range(nSteps):
			# train

			for j in range(batchSize):
				fBrig = maxBrig * np.float_power(-1, np.random.rand() < 0.5) * np.random.rand()
				fCont = 1 + maxCont * np.float_power(-1, np.random.rand() < 0.5) * np.random.rand()
				image =np.zeros((imSize, imSize, nChannels))
				for iChan in range(nChannels):
					image[:,:,iChan]= Train[permT[jT + j], :, :, math.floor(nAug*np.random.rand()), iChan] * fCont + fBrig
					# image[:, :, iChan] = Train[permT[jT + j], :, :, 0,iChan]
				batchData[j, :, :, :] =  image

				batchLabels[j, :, :, :] = LTrain[permT[jT + j], :, :, :]
				batchWeights[j, :, :, :] = WTrain[permT[jT + j], :, :, :]
			summary, _ = sess.run([merged, optOp], feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels,
															  tfWeights: batchWeights, UNet2D.tfTraining: 1})
			jT = jT + batchSize
			if jT > (nTrain - batchSize - 1):
				jT = 0
				np.random.shuffle(permT)
				epochCounter = epochCounter + 1
			if np.mod(i, 20) == 0:
				trainWriter.add_summary(summary, i)

			# validation
			for j in range(batchSize):
				image = np.zeros((imSize, imSize, nChannels))
				image[:, :, 0] = Valid[permV[jV + j], :, :, math.floor(nAug*np.random.rand()), 0]
				# image[:, :, 1] = Valid[permV[jV + j], :, :, 0, 1]
				batchData[j, :, :, :] = image
				batchLabels[j, :, :, :] = LValid[permV[jV + j], :, :, :]
				batchWeights[j, :, :, :] = WValid[permV[jV + j], :, :, :]
			summary, es = sess.run([merged, errors], feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels,
																tfWeights: batchWeights, UNet2D.tfTraining: 0})
			jV = jV + batchSize
			if jV > (nValid - batchSize - 1):
				jV = 0
				np.random.shuffle(permV)
			if np.mod(i, 20) == 0:
				validWriter.add_summary(summary, i)

			e = np.mean(es)
			print('step %05d, e: %f' % (i, e) + ', epoch: ' + str(epochCounter))

			if i == 0:
				if restoreVariables:
					lowestError = e
				else:
					lowestError = np.inf

			if np.mod(i, 50) == 0 and e < lowestError:
				lowestError = e
				print("Model saved in file: %s" % saver.save(sess, outModelPath))

		# --------------------------------------------------
		# save hyper-parameters, clean-up
		# --------------------------------------------------

		saveData(UNet2D.hp, pathjoin(modelPath, 'hp.data'))

		trainWriter.close()
		validWriter.close()
		sess.close()


		# --------------------------------------------------
		# test
		# --------------------------------------------------
		os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuIndex
		tf.reset_default_graph()
		variablesPath = pathjoin(modelPath, 'model.ckpt')
		outPMPath = pmPath

		hp = loadData(pathjoin(modelPath, 'hp.data'))
		UNet2D.setupWithHP(hp)
		saver = tf.train.Saver()
		sess = tf.Session(config=tf.ConfigProto(
			allow_soft_placement=True))  # config parameter needed to save variables when using GPU
		saver.restore(sess, variablesPath)
		print("Model restored.")

		if not os.path.exists(outPMPath):
			os.makedirs(outPMPath)

		for iAug in range(nAug):
			for i in range(nTest):
				j = np.mod(i, batchSize)
				image = np.zeros((imSize, imSize, nChannels))
				image[:,:,0] = Test[i, :, :, iAug, 0]
				# image[:, :, 1] = Test[i, :, :, 0, 1]
				batchData[j, :, :, :] = image
				batchLabels[j, :, :, :] = LTest[i, :, :, :]

				if j == batchSize - 1 or i == nTest - 1:

					output = sess.run(UNet2D.nn,
									  feed_dict={UNet2D.tfData: batchData, UNet2D.tfTraining: 0})

					for k in range(j + 1):
						pm = output[k, :, :, 2]
						gt = batchLabels[k, :, :, 2]
						im = np.sqrt(normalize(batchData[k, :, :, 0]))
						imwrite(np.uint8(255 * np.concatenate((im, np.concatenate((pm, gt), axis=1)), axis=1)),
								'%s/I%05d_%d_Nuc.png' % (outPMPath, i - j + k + 1,iAug))

					for k in range(j + 1):
						pm = output[k, :, :, 1]
						gt = batchLabels[k, :, :, 1]
						im = np.sqrt(normalize(batchData[k, :, :, 0]))
						imwrite(np.uint8(255 * np.concatenate((im, np.concatenate((pm, gt), axis=1)), axis=1)),
								'%s/I%05d_%d_Con.png' % (outPMPath, i - j + k + 1,iAug))



	def deploy(imPath, nImages, modelPath, pmPath, gpuIndex, pmIndex):
		os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuIndex
		tf.reset_default_graph()
		variablesPath = pathjoin(modelPath, 'model.ckpt')
		outPMPath = pmPath

		hp = loadData(pathjoin(modelPath, 'hp.data'))
		UNet2D.setupWithHP(hp)

		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']
		nClasses = UNet2D.hp['nClasses']

		# --------------------------------------------------
		# data
		# --------------------------------------------------

		Data = np.zeros((nImages, imSize, imSize, nChannels))

		datasetMean = loadData(pathjoin(modelPath, 'datasetMean.data'))
		datasetStDev = loadData(pathjoin(modelPath, 'datasetStDev.data'))

		for iSample in range(0, nImages):
			path = '%s/I%05d_Img.tif' % (imPath, iSample)
			for iChan in range(nChannels):
				im = im2double(skio.imread(path,img_num = iChan))
				Data[iSample, :, :, iChan] = (im - datasetMean) / datasetStDev

		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		sess = tf.Session(config=tf.ConfigProto(
			allow_soft_placement=True))  # config parameter needed to save variables when using GPU

		saver.restore(sess, variablesPath)
		print("Model restored.")

		# --------------------------------------------------
		# deploy
		# --------------------------------------------------

		batchData = np.zeros((batchSize, imSize, imSize, nChannels))

		if not os.path.exists(outPMPath):
			os.makedirs(outPMPath)

		for i in range(nImages):
			print(i, nImages)

			j = np.mod(i, batchSize)

			batchData[j, :, :, :] = Data[i, :, :, :]

			if j == batchSize - 1 or i == nImages - 1:

				output = sess.run(UNet2D.nn, feed_dict={UNet2D.tfData: batchData, UNet2D.tfTraining: 0})

				for k in range(j + 1):
					pm = output[k, :, :, pmIndex]
					im = np.sqrt(normalize(batchData[k, :, :, 0]))
					# imwrite(np.uint8(255*np.concatenate((im,pm),axis=1)),'%s/I%05d.png' % (outPMPath,i-j+k+1))
					imwrite(np.uint8(255 * im), '%s/I%05d_Im.png' % (outPMPath, i - j + k + 1))
					imwrite(np.uint8(255 * pm), '%s/I%05d_PM.png' % (outPMPath, i - j + k + 1))

		# --------------------------------------------------
		# clean-up
		# --------------------------------------------------

		sess.close()

	def singleImageInferenceSetup(modelPath, gpuIndex, mean, std):
		variablesPath = pathjoin(modelPath, 'model.ckpt')

		hp = loadData(pathjoin(modelPath, 'hp.data'))
		UNet2D.setupWithHP(hp)
		if mean == -1:
			UNet2D.DatasetMean = loadData(pathjoin(modelPath, 'datasetMean.data'))
		else:
			UNet2D.DatasetMean = mean

		if std == -1:
			UNet2D.DatasetStDev = loadData(pathjoin(modelPath, 'datasetStDev.data'))
		else:
			UNet2D.DatasetStDev = std
		print(UNet2D.DatasetMean)
		print(UNet2D.DatasetStDev)

		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		UNet2D.Session = tf.Session(config=tf.ConfigProto(
			allow_soft_placement=True))  # config parameter needed to save variables when using GPU

		saver.restore(UNet2D.Session, variablesPath)
		print("Model restored.")

	def singleImageInferenceCleanup():
		UNet2D.Session.close()

	def singleImageInference(image, mode, pmIndex):
		print('Inference...')

		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']

		PI2D.setup(image, imSize, int(imSize / 8), mode)
		PI2D.createOutput(1)

		batchData = np.zeros((batchSize, imSize, imSize, nChannels))
		for i in range(PI2D.NumPatches):
			j = np.mod(i, batchSize)
			P = (PI2D.getPatch(i) - UNet2D.DatasetMean) / UNet2D.DatasetStDev
			for iChan in range(nChannels):
				batchData[j, :, :, iChan] = P#[iChan, :, :]
			if j == batchSize - 1 or i == PI2D.NumPatches - 1:
				output = UNet2D.Session.run(UNet2D.nn, feed_dict={UNet2D.tfData: batchData, UNet2D.tfTraining: 0})
				for k in range(j + 1):
					pm = output[k, :, :, pmIndex]
					PI2D.patchOutput(i - j + k, pm)
		# PI2D.patchOutput(i-j+k,normalize(imgradmag(PI2D.getPatch(i-j+k),1)))

		return PI2D.getValidOutput()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("imagePath", help="path to the .tif file")
	parser.add_argument("--model", help="type of model. For example, nuclei vs cytoplasm", default='nucleiDAPI1-5')
	parser.add_argument("--outputPath", help="output path of probability map")
	parser.add_argument("--channel", help="channel to perform inference on", nargs='+', default=[0])
	# parser.add_argument("--channel2", help="channel2 to perform inference on", type=int, default=-1)
	parser.add_argument("--classOrder", help="background, contours, foreground", type=int, nargs='+', default=-1)
	parser.add_argument("--mean", help="mean intensity of input image. Use -1 to use model", type=float, default=-1)
	parser.add_argument("--std", help="mean standard deviation of input image. Use -1 to use model", type=float,
						default=-1)
	parser.add_argument("--scalingFactor", help="factor by which to increase/decrease image size by", type=float,
						default=1)
	parser.add_argument("--stackOutput", help="save probability maps as separate files", action='store_true')
	parser.add_argument("--GPU", help="explicitly select GPU", type=int, default=-1)
	parser.add_argument("--outlier",
						help="map percentile intensity to max when rescaling intensity values. Max intensity as default",
						type=float, default=-1)
	args = parser.parse_args()

	logPath = ''
	scriptPath = os.path.dirname(os.path.realpath(__file__))
	modelPath = os.path.join(scriptPath, 'models', args.model)
	# modelPath = os.path.join(scriptPath, 'models/cytoplasmINcell')
	# modelPath = os.path.join(scriptPath, 'cytoplasmZeissNikon')
	pmPath = ''
	if os.system('nvidia-smi') == 0:
		if args.GPU == -1:
			print("automatically choosing GPU")
			GPU = GPUselect.pick_gpu_lowest_memory()
		else:
			GPU = args.GPU
		print('Using GPU ' + str(GPU))

	else:
		if sys.platform == 'win32':  # only 1 gpu on windows
			if args.GPU == -1:
				GPU = 0
				print('using default GPU')
			else:
				GPU = args.GPU
			print('Using GPU ' + str(GPU))
		else:
			GPU = 0
			print('Using CPU')
	os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % GPU
	UNet2D.singleImageInferenceSetup(modelPath, GPU, args.mean, args.std)
	nClass = UNet2D.hp['nClasses']
	imagePath = args.imagePath
	dapiChannel = args.channel[0]
	channel = args.channel[0]
	# if len(args.channel) > 1:
	# 	channel = args.channel[0]
	# else:
	# 	channel = args.channel
	print('Using channel ' + str(channel))
	dsFactor = args.scalingFactor
	parentFolder = os.path.dirname(os.path.dirname(imagePath))
	fileName = os.path.basename(imagePath)
	fileNamePrefix = fileName.split(os.extsep, 1)
	print(fileName)
	fileType = fileNamePrefix[1]

	if fileType == 'ome.tif' or fileType == 'btf':
		I = skio.imread(imagePath, img_num=int(channel), plugin='tifffile')
	elif fileType == 'tif':
		I = tifffile.imread(imagePath, key=int(channel))
	elif fileType == 'czi':
		with czifile.CziFile(imagePath) as czi:
			image = czi.asarray()
			I = image[0, 0, int(channel), 0, 0, :, :, 0]
	elif fileType == 'nd2':
		with ND2Reader(imagePath) as fullStack:
			I = fullStack[int(channel[iChan])]
	hsize = int((float(I.shape[0]) * float(dsFactor)))
	vsize = int((float(I.shape[1]) * float(dsFactor)))
	I = resize(I, (hsize, vsize))
	cells = np.zeros((I.shape[0], I.shape[1]))
	if args.outlier == -1:
		maxLimit = np.max(I)
	else:
		maxLimit = np.percentile(I, args.outlier)
	I = im2double(sk.rescale_intensity(I, in_range=(np.min(I), maxLimit), out_range=(0, 0.983)))
	cells = I
	rawI = cells
	if args.classOrder == -1:
		args.classOrder = range(nClass)

	rawI = im2double(rawI) / np.max(im2double(rawI))
	if not args.outputPath:
		args.outputPath = parentFolder + '//probability_maps'

	if not os.path.exists(args.outputPath):
		os.makedirs(args.outputPath)

	append_kwargs = {
		'bigtiff': True,
		'metadata': None,
		'append': True,
	}
	save_kwargs = {
		'bigtiff': True,
		'metadata': None,
		'append': False,
	}
	if args.stackOutput:
		slice = 0
		for iClass in args.classOrder[::-1]:
			PM = np.uint8(255 * UNet2D.singleImageInference(cells, 'accumulate',
															iClass))  # backwards in order to align with ilastik...
			PM = resize(PM, (rawI.shape[0], rawI.shape[1]))
			if slice == 0:
				skimage.io.imsave(
					args.outputPath + '//' + fileNamePrefix[0] + '_Probabilities_' + str(dapiChannel) + '.tif',
					np.uint8(255 * PM), **save_kwargs)
			else:
				skimage.io.imsave(
					args.outputPath + '//' + fileNamePrefix[0] + '_Probabilities_' + str(dapiChannel) + '.tif',
					np.uint8(255 * PM), **append_kwargs)
			if slice == 1:
				save_kwargs['append'] = False
				skimage.io.imsave(args.outputPath + '//' + fileNamePrefix[0] + '_Preview_' + str(dapiChannel) + '.tif',
								  np.uint8(255 * PM), **save_kwargs)
				skimage.io.imsave(args.outputPath + '//' + fileNamePrefix[0] + '_Preview_' + str(dapiChannel) + '.tif',
								  np.uint8(255 * rawI), **append_kwargs)
			slice = slice + 1

	else:
		contours = np.uint8(255 * UNet2D.singleImageInference(cells, 'accumulate', args.classOrder[1]))
		contours = resize(contours, (rawI.shape[0], rawI.shape[1]))
		skimage.io.imsave(args.outputPath + '//' + fileNamePrefix[0] + '_ContoursPM_' + str(dapiChannel) + '.tif',
						  np.uint8(255 * contours), **save_kwargs)
		skimage.io.imsave(args.outputPath + '//' + fileNamePrefix[0] + '_ContoursPM_' + str(dapiChannel) + '.tif',
						  np.uint8(255 * rawI), **append_kwargs)
		del contours
		nuclei = np.uint8(255 * UNet2D.singleImageInference(cells, 'accumulate', args.classOrder[2]))
		nuclei = resize(nuclei, (rawI.shape[0], rawI.shape[1]))
		skimage.io.imsave(args.outputPath + '//' + fileNamePrefix[0] + '_NucleiPM_' + str(dapiChannel) + '.tif',
						  np.uint8(255 * nuclei), **save_kwargs)
		del nuclei
	UNet2D.singleImageInferenceCleanup()




	# logPath = 'D:\\LSP\\UNet\\TuuliaLPTBdapiTFv2\\TFLogs'
	# modelPath = 'D:\\LSP\\UNet\\TuuliaLPTBdapiTFv2'
	# pmPath = 'D:\\LSP\\UNet\\TuuliaLPTBdapiTFv2\\TFProbMaps'
	# pmPath1 = 'D:\\LSP\\UNet\\LPTCGSdapiRTAug6-18-16\\TFProbMaps1'

	# ----- test 1 -----

	# imPath = 'D:\\LSP\\cycif\\LPTBdapiRTAug64'
	# validPath = 'D:\\LSP\\cycif\\LPTBdapiRTAug64\\valid'
	# testPath = 'D:\\LSP\\cycif\\LPTBdapiRTAug64\\test'
	#
	# # UNet2D.setup(64, 1, 3, 80, 2, 2, 3, 0, 0.03, 4, 32)  # 64 0.0001 4
	# # UNet2D.train(imPath, validPath, testPath, logPath, modelPath, pmPath,  3401, 120, 196, False, 20000, 0, 2)
	# UNet2D.deploy(testPath, 196, modelPath, pmPath, 0, 1)

	# UNet2D.singleImageInferenceSetup(modelPath, 1)
	# imagePath = 'Y:\\sorger\\data\\RareCyte\\Tuulia\\mcmicro\\Topacio\\01092020'
	# sampleList = glob.glob(imagePath + '\\*')
	# dapiChannel = 0
	# # laminChannel = 33
	# dsFactor =1
	# for iSample in sampleList:
	# 	fileList = glob.glob(iSample + '\\registration\\*.ome.tif')
	# 	print(fileList)
	# 	for iFile in fileList:
	# 		fileName = os.path.basename(iFile)
	# 		fileNamePrefix = fileName.split(os.extsep, 1)
	# 		#I = tifffile.imread(iFile, key=dapiChannel)
	# 		dapi = skio.imread(iFile, key=dapiChannel,img_num=dapiChannel)
	# 		# lamin = skio.imread(iFile, key=laminChannel, img_num=laminChannel)
	# 		rawI = dapi
	# 		hsize = int((float(dapi.shape[0]) * float(dsFactor)))
	# 		vsize = int((float(dapi.shape[1]) * float(dsFactor)))
	# 		dapi = cv2.resize(dapi, (vsize, hsize), interpolation=cv2.INTER_CUBIC)
	# 		# lamin = cv2.resize(lamin, (vsize, hsize), interpolation=cv2.INTER_CUBIC)
	#
	# 		dapi=im2double(dapi)
	# 		# lamin = im2double(lamin)
	# 		dapi = im2double(sk.rescale_intensity(dapi, in_range=(np.percentile(dapi,1), np.percentile(dapi,99.9)), out_range=(0, 0.983)))
	#
	#
	# 		rawI = im2double(rawI)/np.max(im2double(rawI))
	# 		outputPath = iSample + '//probmapsSaturated'
	# 		if not os.path.exists(outputPath):
	# 			os.makedirs(outputPath)
	# 		K = np.zeros((2,rawI.shape[0],rawI.shape[1]))
	# 		contours = UNet2D.singleImageInference(dapi,'accumulate',1)
	# 		hsize = rawI.shape[0]
	# 		vsize = rawI.shape[1]
	# 		contours = resize(contours,(hsize,vsize))
	# 		# contours = cv2.resize(contours,  (vsize, hsize), interpolation=cv2.INTER_CUBIC)
	# 		K[1,:,:] = rawI
	# 		K[0,:,:] = contours
	# 		tifwrite(np.uint8(255 * K),
	# 				 outputPath + '//' + fileNamePrefix[0] + '_ContoursPM_' + str(dapiChannel + 1) + '.tif')
	# 		del K
	# UNet2D.singleImageInferenceCleanup()
