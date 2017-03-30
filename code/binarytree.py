# implementing an n-ary tree lstm
# 3 layer deep lstm
# 1st layer takes i/p 0,2,4,...18 and o/p h1,h2,..h5
# 2nd layer takes i/p 1,5,9,13,17 with h1,h2,h3,h4,h5 and o/p hh1,hh2,hh3,hh4,hh5
# 3rd layer takes remaining i/p with hh1,...hh5 and o/p 5 outputs
# IMPLEMENTATION IN PROGRESS

import tensorflow as tf

def create(model, config):
	dim_v, dim_i, dim_d, dim_t, dim_b, dim_n, dim_c = config.getint('vocabsize'), config.getint('wvecsize'), config.getint('depth'), config.getint('steps'), config.getint('batch'), config.getint('deepness'), config.getint('classes')
	lrate_ms, dstep_ms, drate_ms, optim_ms = config.getfloat('mslrate'), config.getint('msdstep'), config.getfloat('msdrate'), getattr(tf.train, config.get('msoptim'))
	lrate_ce, dstep_ce, drate_ce, optim_ce = config.getfloat('celrate'), config.getint('cedstep'), config.getfloat('cedrate'), getattr(tf.train, config.get('ceoptim'))

	with tf.name_scope('embedding'):
		model['We'] = tf.Variable(tf.truncated_normal([dim_v, dim_i], stddev = 1.0 / dim_i), name = 'We')
		model['Be'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'Be')

	with tf.name_scope('plstm'):
		with tf.name_scope('input'):
			for ii in xrange(dim_t):
				model['pxi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'pxi_%i' %ii)
				model['px_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['pxi_%i' %ii]), model['Be'], name = 'px_%i' %ii)

		with tf.name_scope('label'):
			for ii in xrange(dim_t):
				model['pyi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'pyi_%i' %ii)
				model['py_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['pyi_%i' %ii]), model['Be'], name = 'py_%i' %ii)

		for i in xrange(dim_d):
			dim_t_cur=dim_t/(2**i)
			with tf.name_scope('input_%i' %i):
				for ii in xrange(dim_t_cur):
					model['px_%i_%i' %(i, ii)] = model['px_%i' %ii] if i == 0 else model['ph_%i_%i' %(i - 1, ii)] # input

			with tf.name_scope('inputgate_%i' %i):
				model['pWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWi_%i' %i)
				model['pUi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pUi_%i' %i)
				model['pBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBi_%i' %i)
				for ii in xrange(dim_t):
					model['pch_%i_%i' %(i, ii)] = tf.zeros([1,dim_i], tf.int32) if i == 0 else tf.add(tf.matmul(model['pUi_%i']))
					model['pi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWi_%i' %i]), model['pBi_%i' %i]), name = 'pi_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['pWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWf_%i' %i)
				model['pBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBf_%i' %i)
				for ii in xrange(dim_t):
					model['pf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWf_%i' %i]), model['pBf_%i' %i]), name = 'pf_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['pWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWo_%i' %i)
				model['pBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBo_%i' %i)
				for ii in xrange(dim_t):
					model['po_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWo_%i' %i]), model['pBo_%i' %i]), name = 'po_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['pWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWc_' + str(i))
				model['pBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBc_' + str(i))
				for ii in xrange(dim_t):
					model['pcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'pcc_%i_%i' %(i, ii)) if ii == 0 else model['pc_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['pc_%i_%i' %(i, ii)] = tf.where(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['pcc_%i_%i' %(i, ii)], tf.add(tf.multiply(model['pf_%i_%i' %(i, ii)], model['pcc_%i_%i' %(i, ii)]), tf.multiply(model['pi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWc_%i' %i]), model['pBc_%i' %i])))), name = 'pc_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['pWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWz_%i' %i)
				model['pBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBz_%i' %i)
				for ii in xrange(dim_t):
					model['pz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['pc_%i_%i' %(i, ii)], model['pWz_%i' %i]), model['pBz_%i' %i], name = 'pz_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['ph_%i_%i' %(i, ii)] = tf.multiply(model['po_%i_%i' %(i, ii)], tf.nn.tanh(model['pz_%i_%i' %(i, ii)]), name = 'ph_%i_%i' %(i, ii))
				
