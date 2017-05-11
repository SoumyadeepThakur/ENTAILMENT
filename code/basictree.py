# implementing a basic tree lstm
# 1st level: inputs x0,x3,x4,x7,x8,x11,x12,x15,x16,x19
# 2nd level: inputs x1,x5,x9,x13,x17
# 3rd level: remaining inputs


import tensorflow as tf

def create(model, config):
	dim_v, dim_i, dim_d, dim_t, dim_b, dim_n, dim_c = config.getint('vocabsize'), config.getint('wvecsize'), config.getint('depth'), config.getint('steps'), config.getint('batch'), config.getint('deepness'), config.getint('classes')
	lrate_ms, dstep_ms, drate_ms, optim_ms = config.getfloat('mslrate'), config.getint('msdstep'), config.getfloat('msdrate'), getattr(tf.train, config.get('msoptim'))
	lrate_ce, dstep_ce, drate_ce, optim_ce = config.getfloat('celrate'), config.getint('cedstep'), config.getfloat('cedrate'), getattr(tf.train, config.get('ceoptim'))

	with tf.name_scope('embedding'):
		model['We'] = tf.Variable(tf.truncated_normal([dim_v, dim_i], stddev = 1.0 / dim_i), name = 'We')
		model['Be'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'Be')

	#premise lstm
	with tf.name_scope('plstm'):
		with tf.name_scope('input'):
			for ii in xrange(dim_t):
				model['pxi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'pxi_%i' %ii)
				model['px_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['pxi_%i' %ii]), model['Be'], name = 'px_%i' %ii)

		with tf.name_scope('label'):
			for ii in xrange(dim_t):
				model['pyi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'pyi_%i' %ii)
				model['py_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['pyi_%i' %ii]), model['Be'], name = 'py_%i' %ii)

	with tf.name_scope('input'):
		for i in xrange(dim_t):
			model['pFx_%i' %i] = model['px_%i' %i]
			model['pBx_%i' %i] = model['px_%i' %i]

	# build the 3 gates i/p forget and o/p
	with tf.name_scope('inputgate'):
		model['pWi'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWi')
		model['pBi'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBi')
		for i in xrange(dim_t):
			model['pi_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['pFx_%i' %i], model['pWi']), model['pBi']), name = 'pi_%i' %i)

	with tf.name_scope('forgetgate'):
		model['pWf'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWf')
		model['pBf'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBf')
		for i in xrange(dim_t):
			model['pf_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['pFx_%i' %i], model['pWf']), model['pBf']), name = 'pf_%i' %i)

	with tf.name_scope('outputgate'):
		model['pWo'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWo')
		model['pBo'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBo')
		for i in xrange(dim_t):
			model['po_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['pFx_%i' %i], model['pWo_%i' %i]), model['pBo_%i' %i]), name = 'po_%i' %i)

	with tf.name_scope('cellstate'):
		model['pWc'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWc')
		model['pBc'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBc')
		for i in xrange(dim_t):
			if i % 4 == 1:
				model['pf_prev_%i' %i] = tf.add(tf.multiply(model['pf_%i' %(i-1)], model['pc_%i' %(i-1)]), tf.multiply(model['pf_%i' %(i+2)], model['pc_%i' %(i+2)]), name = 'pf_prev_%i' %i)
			elif i % 4 == 2:
				model['pf_prev_%i' %i] = tf.add(tf.multiply(model['pf_%i' %(i-1)], model['pc_%i' %(i-1)]), tf.multiply(model['pf_%i' %((i+3)%dim_t)], model['pc_%i' %((i+3)%dim_t)]), name = 'pf_prev_%i' %i)
			else:
				model['pf_prev_%i' %i] = #add something here
			model['pcc_%i' %i] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'pcc_%i' %i) # consider starting with all zeros
			#to edit
			model['pc_%i' %i]=tf.where(tf.equals(model['pxi_%i' %i], tf.zeros([dim_b], tf.int32)), model['pcc_%i' %i], tf.add(tf.multiply(model['pf_%i' %i], model['pcc_%i' %i]), tf.multiply(model['pFi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['pFx_%i_%i' %(i, ii)], model['pFWc_%i' %i]), model['pFBc_%i' %i])))), name = 'pFc_%i_%i' %(i, ii)) #consider the 2 children input states

	#insert hypothesis lstm  here

	#softmax and crossentropy
	with tf.name_scope('classification'):
		with tf.name_scope('label'):
			model['clabel'] = tf.placeholder(tf.float32, [dim_b, dim_c], name = 'clabel')
 
		for i in xrange(dim_n):
			with tf.name_scope('layer_%i' %i):
				model['cW_%i' %i] = tf.Variable(tf.truncated_normal([4 * dim_i, 4 * dim_i], stddev = 0.25 / dim_i), name = 'cW_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([4 * dim_i, dim_c], stddev = 1.0 / dim_c), name = 'cW_%i' %i)
				model['cB_%i' %i] = tf.Variable(tf.truncated_normal([1, 4 * dim_i], stddev = 0.25 / dim_i), name = 'cB_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([1, dim_c], stddev = 1.0 / dim_c), name = 'cB_%i' %i)
				model['cx_%i' %i] = tf.concat(axis=1, values=[model['ph_%i' %(dim_t - 1)], model['ph_%i' %(0)], model['hh_%i' %(dim_t - 1)], model['hh_%i' %(0)]], name = 'cx_%i' %i) if i == 0 else model['cy_%i' %(i - 1)]
				model['cy_%i' %i] = tf.add(tf.matmul(model['cx_%i' %i], model['cW_%i' %i]), model['cB_%i' %i], name = 'cy_%i' %i)

		with tf.name_scope('output'):
			model['output'] = tf.nn.softmax(model['cy_%i' %(dim_n - 1)], name = 'output')

		with tf.name_scope('crossentropy'):
			model['cce'] = tf.reduce_sum(-tf.multiply(model['clabel'], tf.log(model['output'])), name = 'cce')
			model['scce'] = tf.summary.scalar(model['cce'].name, model['cce'])

	model['gsms'] = tf.Variable(0, trainable = False, name = 'gsms')
	model['lrms'] = tf.train.exponential_decay(lrate_ms, model['gsms'], dstep_ms, drate_ms, staircase = False, name = 'lrms')
	model['tms'] = optim_ms(model['lrms']).minimize(model['pFms'] + model['pBms'] + model['hFms'] + model['hBms'], global_step = model['gsms'], name = 'tms')

	model['gsce'] = tf.Variable(0, trainable = False, name = 'gsce')
	model['lrce'] = tf.train.exponential_decay(lrate_ce, model['gsce'], dstep_ce, drate_ce, staircase = False, name = 'lrce')
	model['tce'] = optim_ce(model['lrce']).minimize(model['cce'], global_step = model['gsce'], name = 'tce')

	return model			
