import numpy as np
from scipy import misc
import tensorflow as tf
import shutil
import scipy.io as sio
import os,fnmatch,glob
import skimage.exposure as sk

import sys
sys.path.insert(0, 'C:\\Users\\Clarence\\Documents\\UNet code\\ImageScience')
from toolbox.imtools import *
from toolbox.ftools import *
from toolbox.PartitionOfImage import PI2D


def concat3(lst):
		return tf.concat(lst,3)

class UNet2D:
	hp = None # hyper-parameters
	nn = None # network
	tfTraining = None # if training or not (to handle batch norm)
	tfData = None # data placeholder
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

	def setup(imSize,nChannels,nClasses,nOut0,featMapsFact,downSampFact,kernelSize,nExtraConvs,stdDev0,nDownSampLayers,batchSize):
		UNet2D.hp = {'imSize':imSize,
					 'nClasses':nClasses,
					 'nChannels':nChannels,
					 'nExtraConvs':nExtraConvs,
					 'nLayers':nDownSampLayers,
					 'featMapsFact':featMapsFact,
					 'downSampFact':downSampFact,
					 'ks':kernelSize,
					 'nOut0':nOut0,
					 'stdDev0':stdDev0,
					 'batchSize':batchSize}

		nOutX = [UNet2D.hp['nChannels'],UNet2D.hp['nOut0']]
		dsfX = []
		for i in range(UNet2D.hp['nLayers']):
			nOutX.append(nOutX[-1]*UNet2D.hp['featMapsFact'])
			dsfX.append(UNet2D.hp['downSampFact'])


		# --------------------------------------------------
		# downsampling layer
		# --------------------------------------------------

		with tf.name_scope('placeholders'):
			UNet2D.tfTraining = tf.placeholder(tf.bool, name='training')
			UNet2D.tfData = tf.placeholder("float", shape=[None,UNet2D.hp['imSize'],UNet2D.hp['imSize'],UNet2D.hp['nChannels']],name='data')

		def down_samp_layer(data,index):
			with tf.name_scope('ld%d' % index):
				ldXWeights1 = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index], nOutX[index+1]], stddev=stdDev0),name='kernel1')
				ldXWeightsExtra = []
				for i in range(nExtraConvs):
					ldXWeightsExtra.append(tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index+1], nOutX[index+1]], stddev=stdDev0),name='kernelExtra%d' % i))
				
				c00 = tf.nn.conv2d(data, ldXWeights1, strides=[1, 1, 1, 1], padding='SAME')
				for i in range(nExtraConvs):
					c00 = tf.nn.conv2d(tf.nn.relu(c00), ldXWeightsExtra[i], strides=[1, 1, 1, 1], padding='SAME')

				ldXWeightsShortcut = tf.Variable(tf.truncated_normal([1, 1, nOutX[index], nOutX[index+1]], stddev=stdDev0),name='shortcutWeights')
				shortcut = tf.nn.conv2d(data, ldXWeightsShortcut, strides=[1, 1, 1, 1], padding='SAME')

				bn = tf.layers.batch_normalization(tf.nn.relu(c00+shortcut), training=UNet2D.tfTraining)

				return tf.nn.max_pool(bn, ksize=[1, dsfX[index], dsfX[index], 1], strides=[1, dsfX[index], dsfX[index], 1], padding='SAME',name='maxpool')

		# --------------------------------------------------
		# bottom layer
		# --------------------------------------------------

		with tf.name_scope('lb'):
			lbWeights1 = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[UNet2D.hp['nLayers']], nOutX[UNet2D.hp['nLayers']+1]], stddev=stdDev0),name='kernel1')
			def lb(hidden):
				return tf.nn.relu(tf.nn.conv2d(hidden, lbWeights1, strides=[1, 1, 1, 1], padding='SAME'),name='conv')

		# --------------------------------------------------
		# downsampling
		# --------------------------------------------------

		with tf.name_scope('downsampling'):    
			dsX = []
			dsX.append(UNet2D.tfData)

			for i in range(UNet2D.hp['nLayers']):
				dsX.append(down_samp_layer(dsX[i],i))

			b = lb(dsX[UNet2D.hp['nLayers']])

		# --------------------------------------------------
		# upsampling layer
		# --------------------------------------------------

		def up_samp_layer(data,index):
			with tf.name_scope('lu%d' % index):
				luXWeights1    = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index+1], nOutX[index+2]], stddev=stdDev0),name='kernel1')
				luXWeights2    = tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index]+nOutX[index+1], nOutX[index+1]], stddev=stdDev0),name='kernel2')
				luXWeightsExtra = []
				for i in range(nExtraConvs):
					luXWeightsExtra.append(tf.Variable(tf.truncated_normal([UNet2D.hp['ks'], UNet2D.hp['ks'], nOutX[index+1], nOutX[index+1]], stddev=stdDev0),name='kernel2Extra%d' % i))
				
				outSize = UNet2D.hp['imSize']
				for i in range(index):
					outSize /= dsfX[i]
				outSize = int(outSize)

				outputShape = [UNet2D.hp['batchSize'],outSize,outSize,nOutX[index+1]]
				us = tf.nn.relu(tf.nn.conv2d_transpose(data, luXWeights1, outputShape, strides=[1, dsfX[index], dsfX[index], 1], padding='SAME'),name='conv1')
				cc = concat3([dsX[index],us]) 
				cv = tf.nn.relu(tf.nn.conv2d(cc, luXWeights2, strides=[1, 1, 1, 1], padding='SAME'),name='conv2')
				for i in range(nExtraConvs):
					cv = tf.nn.relu(tf.nn.conv2d(cv, luXWeightsExtra[i], strides=[1, 1, 1, 1], padding='SAME'),name='conv2Extra%d' % i)
				return cv

		# --------------------------------------------------
		# final (top) layer
		# --------------------------------------------------

		with tf.name_scope('lt'):
			ltWeights1    = tf.Variable(tf.truncated_normal([1, 1, nOutX[1], nClasses], stddev=stdDev0),name='kernel')
			def lt(hidden):
				return tf.nn.conv2d(hidden, ltWeights1, strides=[1, 1, 1, 1], padding='SAME',name='conv')


		# --------------------------------------------------
		# upsampling
		# --------------------------------------------------

		with tf.name_scope('upsampling'):
			usX = []
			usX.append(b)

			for i in range(UNet2D.hp['nLayers']):
				usX.append(up_samp_layer(usX[i],UNet2D.hp['nLayers']-1-i))

			t = lt(usX[UNet2D.hp['nLayers']])


		sm = tf.nn.softmax(t,-1)
		UNet2D.nn = sm


	def train(imPath,logPath,modelPath,pmPath,nTrain,nValid,nTest,restoreVariables,nSteps,gpuIndex,testPMIndex):
		os.environ['CUDA_VISIBLE_DEVICES']= '%d' % gpuIndex

		outLogPath = logPath
		trainWriterPath = pathjoin(logPath,'Train')
		validWriterPath = pathjoin(logPath,'Valid')
		outModelPath = pathjoin(modelPath,'model.ckpt')
		outPMPath = pmPath
		
		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']
		nClasses = UNet2D.hp['nClasses']

		# --------------------------------------------------
		# data
		# --------------------------------------------------

		Train = np.zeros((nTrain,imSize,imSize,nChannels))
		Valid = np.zeros((nValid,imSize,imSize,nChannels))
		Test = np.zeros((nTest,imSize,imSize,nChannels))
		LTrain = np.zeros((nTrain,imSize,imSize,nClasses))
		LValid = np.zeros((nValid,imSize,imSize,nClasses))
		LTest = np.zeros((nTest,imSize,imSize,nClasses))

		print('loading data, computing mean / st dev')
		if not os.path.exists(modelPath):
			os.makedirs(modelPath)
		if restoreVariables:
			datasetMean = loadData(pathjoin(modelPath,'datasetMean.data'))
			datasetStDev = loadData(pathjoin(modelPath,'datasetStDev.data'))
		else:
			datasetMean = 0
			datasetStDev = 0
			for iSample in range(nTrain+nValid+nTest):
				I = im2double(tifread('%s/I%05d_Img.tif' % (imPath,iSample)))
				datasetMean += np.mean(I)
				datasetStDev += np.std(I)
			datasetMean /= (nTrain+nValid+nTest)
			datasetStDev /= (nTrain+nValid+nTest)
			saveData(datasetMean, pathjoin(modelPath,'datasetMean.data'))
			saveData(datasetStDev, pathjoin(modelPath,'datasetStDev.data'))

		perm = np.arange(nTrain+nValid+nTest)
		np.random.shuffle(perm)

		for iSample in range(0, nTrain):
			path = '%s/I%05d_Img.tif' % (imPath,perm[iSample])
			im = im2double(tifread(path))
			Train[iSample,:,:,0] = (im-datasetMean)/datasetStDev
			path = '%s/I%05d_Ant.tif' % (imPath,perm[iSample])
			im = tifread(path)
			for i in range(nClasses):
				LTrain[iSample,:,:,i] = (im == i+1)

		for iSample in range(0, nValid):
			path = '%s/I%05d_Img.tif' % (imPath,perm[nTrain+iSample])
			im = im2double(tifread(path))
			Valid[iSample,:,:,0] = (im-datasetMean)/datasetStDev
			path = '%s/I%05d_Ant.tif' % (imPath,perm[nTrain+iSample])
			im = tifread(path)
			for i in range(nClasses):
				LValid[iSample,:,:,i] = (im == i+1)

		for iSample in range(0, nTest):
			path = '%s/I%05d_Img.tif' % (imPath,perm[nTrain+nValid+iSample])
			im = im2double(tifread(path))
			Test[iSample,:,:,0] = (im-datasetMean)/datasetStDev
			path = '%s/I%05d_Ant.tif' % (imPath,perm[nTrain+nValid+iSample])
			im = tifread(path)
			for i in range(nClasses):
				LTest[iSample,:,:,i] = (im == i+1)

		# --------------------------------------------------
		# optimization
		# --------------------------------------------------

		tfLabels = tf.placeholder("float", shape=[None,imSize,imSize,nClasses],name='labels')

		globalStep = tf.Variable(0,trainable=False)
		learningRate0 = 0.01
		decaySteps = 1000
		decayRate = 0.95
		learningRate = tf.train.exponential_decay(learningRate0,globalStep,decaySteps,decayRate,staircase=True)

		with tf.name_scope('optim'):
			loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tfLabels,tf.log(UNet2D.nn)),3))
			updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			# optimizer = tf.train.MomentumOptimizer(1e-3,0.9)
			optimizer = tf.train.MomentumOptimizer(learningRate,0.9)
			# optimizer = tf.train.GradientDescentOptimizer(learningRate)
			with tf.control_dependencies(updateOps):
				optOp = optimizer.minimize(loss,global_step=globalStep)

		with tf.name_scope('eval'):
			error = []
			for iClass in range(nClasses):
				labels0 = tf.reshape(tf.to_int32(tf.slice(tfLabels,[0,0,0,iClass],[-1,-1,-1,1])),[batchSize,imSize,imSize])
				predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(UNet2D.nn,3),iClass)),[batchSize,imSize,imSize])
				correct = tf.multiply(labels0,predict0)
				nCorrect0 = tf.reduce_sum(correct)
				nLabels0 = tf.reduce_sum(labels0)
				error.append(1-tf.to_float(nCorrect0)/tf.to_float(nLabels0))
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
			split0 = tf.slice(UNet2D.nn,[0,0,0,0],[-1,-1,-1,1])
			split1 = tf.slice(UNet2D.nn,[0,0,0,1],[-1,-1,-1,1])
			if nClasses > 2:
				split2 = tf.slice(UNet2D.nn,[0,0,0,2],[-1,-1,-1,1])
			tf.summary.image('pm0',split0)
			tf.summary.image('pm1',split1)
			if nClasses > 2:
				tf.summary.image('pm2',split2)
		merged = tf.summary.merge_all()


		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

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

		batchData = np.zeros((batchSize,imSize,imSize,nChannels))
		batchLabels = np.zeros((batchSize,imSize,imSize,nClasses))
		for i in range(nSteps):
			# train

			perm = np.arange(nTrain)
			np.random.shuffle(perm)

			for j in range(batchSize):
				batchData[j,:,:,:] = Train[perm[j],:,:,:]
				batchLabels[j,:,:,:] = LTrain[perm[j],:,:,:]

			summary,_ = sess.run([merged,optOp],feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels, UNet2D.tfTraining: 1})
			trainWriter.add_summary(summary, i)

			# validation

			perm = np.arange(nValid)
			np.random.shuffle(perm)

			for j in range(batchSize):
				batchData[j,:,:,:] = Valid[perm[j],:,:,:]
				batchLabels[j,:,:,:] = LValid[perm[j],:,:,:]

			summary, es = sess.run([merged, errors],feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels, UNet2D.tfTraining: 0})
			validWriter.add_summary(summary, i)

			e = np.mean(es)
			print('step %05d, e: %f' % (i,e))

			if i == 0:
				if restoreVariables:
					lowestError = e
				else:
					lowestError = np.inf

			if np.mod(i,100) == 0 and e < lowestError:
				lowestError = e
				print("Model saved in file: %s" % saver.save(sess, outModelPath))


		# --------------------------------------------------
		# test
		# --------------------------------------------------

		if not os.path.exists(outPMPath):
			os.makedirs(outPMPath)

		for i in range(nTest):
			j = np.mod(i,batchSize)

			batchData[j,:,:,:] = Test[i,:,:,:]
			batchLabels[j,:,:,:] = LTest[i,:,:,:]
		 
			if j == batchSize-1 or i == nTest-1:

				output = sess.run(UNet2D.nn,feed_dict={UNet2D.tfData: batchData, tfLabels: batchLabels, UNet2D.tfTraining: 0})

				for k in range(j+1):
					pm = output[k,:,:,testPMIndex]
					gt = batchLabels[k,:,:,testPMIndex]
					im = np.sqrt(normalize(batchData[k,:,:,0]))
					imwrite(np.uint8(255*np.concatenate((im,np.concatenate((pm,gt),axis=1)),axis=1)),'%s/I%05d.png' % (outPMPath,i-j+k+1))


		# --------------------------------------------------
		# save hyper-parameters, clean-up
		# --------------------------------------------------

		saveData(UNet2D.hp,pathjoin(modelPath,'hp.data'))

		trainWriter.close()
		validWriter.close()
		sess.close()

	def deploy(imPath,nImages,modelPath,pmPath,gpuIndex,pmIndex):
		os.environ['CUDA_VISIBLE_DEVICES']= '%d' % gpuIndex

		variablesPath = pathjoin(modelPath,'model.ckpt')
		outPMPath = pmPath

		hp = loadData(pathjoin(modelPath,'hp.data'))
		UNet2D.setupWithHP(hp)
		
		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']
		nClasses = UNet2D.hp['nClasses']

		# --------------------------------------------------
		# data
		# --------------------------------------------------

		Data = np.zeros((nImages,imSize,imSize,nChannels))

		datasetMean = loadData(pathjoin(modelPath,'datasetMean.data'))
		datasetStDev = loadData(pathjoin(modelPath,'datasetStDev.data'))

		for iSample in range(0, nImages):
			path = '%s/I%05d_Img.tif' % (imPath,iSample)
			im = im2double(tifread(path))
			Data[iSample,:,:,0] = (im-datasetMean)/datasetStDev

		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

		saver.restore(sess, variablesPath)
		print("Model restored.")

		# --------------------------------------------------
		# deploy
		# --------------------------------------------------

		batchData = np.zeros((batchSize,imSize,imSize,nChannels))

		if not os.path.exists(outPMPath):
			os.makedirs(outPMPath)

		for i in range(nImages):
			print(i,nImages)

			j = np.mod(i,batchSize)

			batchData[j,:,:,:] = Data[i,:,:,:]
		 
			if j == batchSize-1 or i == nImages-1:

				output = sess.run(UNet2D.nn,feed_dict={UNet2D.tfData: batchData, UNet2D.tfTraining: 0})

				for k in range(j+1):
					pm = output[k,:,:,pmIndex]
					im = np.sqrt(normalize(batchData[k,:,:,0]))
					# imwrite(np.uint8(255*np.concatenate((im,pm),axis=1)),'%s/I%05d.png' % (outPMPath,i-j+k+1))
					imwrite(np.uint8(255*im),'%s/I%05d_Im.png' % (outPMPath,i-j+k+1))
					imwrite(np.uint8(255*pm),'%s/I%05d_PM.png' % (outPMPath,i-j+k+1))


		# --------------------------------------------------
		# clean-up
		# --------------------------------------------------

		sess.close()

	def singleImageInferenceSetup(modelPath,gpuIndex):
		#os.environ['CUDA_VISIBLE_DEVICES']= '%d' % gpuIndex

		variablesPath = pathjoin(modelPath,'model.ckpt')

		hp = loadData(pathjoin(modelPath,'hp.data'))
		UNet2D.setupWithHP(hp)

		UNet2D.DatasetMean = loadData(pathjoin(modelPath,'datasetMean.data'))
		UNet2D.DatasetStDev = loadData(pathjoin(modelPath,'datasetStDev.data'))
		print(UNet2D.DatasetMean)
		print(UNet2D.DatasetStDev)

		# --------------------------------------------------
		# session
		# --------------------------------------------------

		saver = tf.train.Saver()
		UNet2D.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

		saver.restore(UNet2D.Session, variablesPath)
		print("Model restored.")

	def singleImageInferenceCleanup():
		UNet2D.Session.close()

	def singleImageInference(image,mode,pmIndex):
		print('Inference...')

		batchSize = UNet2D.hp['batchSize']
		imSize = UNet2D.hp['imSize']
		nChannels = UNet2D.hp['nChannels']

		PI2D.setup(image,imSize,int(imSize/8),mode)
		PI2D.createOutput(nChannels)

		batchData = np.zeros((batchSize,imSize,imSize,nChannels))
		for i in range(PI2D.NumPatches):
			j = np.mod(i,batchSize)
			batchData[j,:,:,0] = (PI2D.getPatch(i)-UNet2D.DatasetMean)/UNet2D.DatasetStDev
			if j == batchSize-1 or i == PI2D.NumPatches-1:
				output = UNet2D.Session.run(UNet2D.nn,feed_dict={UNet2D.tfData: batchData, UNet2D.tfTraining: 0})
				for k in range(j+1):
					pm = output[k,:,:,pmIndex]
					PI2D.patchOutput(i-j+k,pm)
					# PI2D.patchOutput(i-j+k,normalize(imgradmag(PI2D.getPatch(i-j+k),1)))

		return PI2D.getValidOutput()


if __name__ == '__main__':
	logPath = 'C://Users//Clarence//Documents//UNet code//TFLogs'
	modelPath = 'D:\\LSP\\UNet\\tonsil20x1bin1chan\\TFModel - 3class 16 kernels 5ks 2 layers'
	pmPath = 'C://Users//Clarence//Documents//UNet code//TFProbMaps'



	UNet2D.singleImageInferenceSetup(modelPath, 0)
	imagePath = 'D:\\LSP\\cycif\\testsets'
	sampleList = glob.glob(imagePath + '//exemplar-001*')
	dapiChannel = 0
	dsFactor = 1
	for iSample in sampleList:
		fileList = glob.glob(iSample + '//registration//*.tif')
		print(fileList)
		for iFile in fileList:
			fileName = os.path.basename(iFile)
			fileNamePrefix = fileName.split(os.extsep, 1)
			I = tifffile.imread(iFile, key=dapiChannel)
			rawI = I
			hsize = int((float(I.shape[0])*float(dsFactor)))
			vsize = int((float(I.shape[1])*float(dsFactor)))
			I = resize(I,(hsize,vsize))
			I = im2double(sk.rescale_intensity(I, in_range=(np.min(I), np.max(I)), out_range=(0, 0.983)))
			rawI = im2double(rawI)/np.max(im2double(rawI))
			outputPath = iSample + '//prob_maps'
			if not os.path.exists(outputPath):
				os.makedirs(outputPath)
			K = np.zeros((2,rawI.shape[0],rawI.shape[1]))
			contours = UNet2D.singleImageInference(I,'accumulate',1)
			hsize = int((float(I.shape[0]) * float(1/dsFactor)))
			vsize = int((float(I.shape[1]) * float(1/dsFactor)))
			contours = resize(contours, (rawI.shape[0], rawI.shape[1]))
			K[1,:,:] = rawI
			K[0,:,:] = contours
			tifwrite(np.uint8(255 * K),
					 outputPath + '//' + fileNamePrefix[0] + '_ContoursPM_' + str(dapiChannel + 1) + '.tif')
			del K
			K = np.zeros((1, rawI.shape[0], rawI.shape[1]))
			nuclei = UNet2D.singleImageInference(I,'accumulate',2)
			nuclei = resize(nuclei, (rawI.shape[0], rawI.shape[1]))
			K[0, :, :] = nuclei
			tifwrite(np.uint8(255 * K),
					 outputPath + '//' + fileNamePrefix[0] + '_NucleiPM_' + str(dapiChannel + 1) + '.tif')
			del K
	UNet2D.singleImageInferenceCleanup()

