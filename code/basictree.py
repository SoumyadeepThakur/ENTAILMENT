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
			for i in xrange(dim_t):
				model['pxi_%i' %i] = tf.placeholder(tf.int32, [dim_b], name = 'pxi_%i' %i)
				model['px_%i' %i] = tf.add(tf.nn.embedding_lookup(model['We'], model['pxi_%i' %i]), model['Be'], name = 'px_%i' %i)

		with tf.name_scope('label'):
			for i in xrange(dim_t):
				model['pyi_%i' %i] = tf.placeholder(tf.int32, [dim_b], name = 'pyi_%i' %i)
				model['py_%i' %i] = tf.add(tf.nn.embedding_lookup(model['We'], model['pyi_%i' %i]), model['Be'], name = 'py_%i' %i)

		#with tf.name_scope('inputs'):
		#	for i in xrange(dim_t):
		#		model['px_%i' %i] = model['px_%i' %i]

		# build the 3 gates i/p forget and o/p
		with tf.name_scope('inputgate'):
			model['pWi'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWi')
			model['pBi'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBi')
			for i in xrange(dim_t):
				model['pi_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i' %i], model['pWi']), model['pBi']), name = 'pi_%i' %i)
				#print model['pi_%i' %i]

		with tf.name_scope('forgetgate'):
			model['pWf'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWf')
			model['pBf'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBf')
			for i in xrange(dim_t):
				model['pf_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i' %i], model['pWf']), model['pBf']), name = 'pf_%i' %i)

		with tf.name_scope('outputgate'):
			model['pWo'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWo')
			model['pBo'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBo')
			for i in xrange(dim_t):
				model['po_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i' %i], model['pWo']), model['pBo']), name = 'po_%i' %i)

		with tf.name_scope('cellstate'):
			model['pWc'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWc')
			model['pBc'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBc')
			#for i in xrange(dim_t):
			#	if i % 4 == 0 or i % 4 == 3:
			#		model['pf_prev_%i' %i] = tf.to_float(tf.zeros([dim_b], tf.int32, name = 'pf_prev_%i' %i)) #is it ok!?
					 
			#for i in xrange(dim_t):
			#	if i % 4 == 1:
			#		model['pf_prev_%i' %i] = tf.to_float(tf.add(tf.multiply(model['pf_%i' %(i-1)], model['pc_%i' %(i-1)]), tf.multiply(model['pf_%i' %(i+2)], model['pc_%i' %(i+2)]), name = 'pf_prev_%i' %i))

			#for i in xrange(dim_t):
			#	if i % 4 == 2:
			#		model['pf_prev_%i' %i] = tf.to_float(tf.add(tf.multiply(model['pf_%i' %(i-1)], model['pc_%i' %(i-1)]), tf.multiply(model['pf_%i' %((i+3)%dim_t)], model['pc_%i' %((i+3)%dim_t)]), name = 'pf_prev_%i' %i))
			for i in xrange(dim_t):
				model['pcc_%i' %i] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'pcc_%i' %i) # consider starting with all zeros

			for i in xrange(dim_t):
				if i % 4 == 0 or i % 4 == 3:
					model['pc_%i' %i] = tf.where(tf.equal(model['pxi_%i' %i], tf.zeros([dim_b], tf.int32)), model['pcc_%i' %i], tf.multiply(model['pi_%i' %i], tf.nn.tanh(tf.add(tf.matmul(model['px_%i' %i], model['pWc']), model['pBc']))), name = 'pc_%i' %i) #consider the 2 children input states
			for i in xrange(dim_t):
				if i % 4 == 1:
					model['pc_%i' %i] = tf.where(tf.equal(model['pxi_%i' %i], tf.zeros([dim_b], tf.int32)), model['pcc_%i' %i], tf.add(tf.to_float(tf.add(tf.multiply(model['pf_%i' %(i-1)], model['pc_%i' %(i-1)]), tf.multiply(model['pf_%i' %(i+2)], model['pc_%i' %(i+2)]))), tf.multiply(model['pi_%i' %i], tf.nn.tanh(tf.add(tf.matmul(model['px_%i' %i], model['pWc']), model['pBc'])))), name = 'pc_%i' %i) #consider the 2 children input states
			for i in xrange(dim_t):
				if i % 4 == 2:
					model['pc_%i' %i] = tf.where(tf.equal(model['pxi_%i' %i], tf.zeros([dim_b], tf.int32)), model['pcc_%i' %i], tf.add(tf.to_float(tf.add(tf.multiply(model['pf_%i' %(i-1)], model['pc_%i' %(i-1)]), tf.multiply(model['pf_%i' %((i+3)%dim_t)], model['pc_%i' %((i+3)%dim_t)]))), tf.multiply(model['pi_%i' %i], tf.nn.tanh(tf.add(tf.matmul(model['px_%i' %i], model['pWc']), model['pBc'])))), name = 'pc_%i' %i) #consider the 2 children input states

		with tf.name_scope('hidden_%i' %i):
			model['pWz'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWz')
			model['pBz'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBz')
			for i in xrange(dim_t):
				model['pz_%i' %i] = tf.add(tf.matmul(model['pc_%i' %i], model['pWz']), model['pBz'], name = 'pz_%i' %i)

		with tf.name_scope('outputs'):
			for i in xrange(dim_t):
				model['ph_%i' %i] = tf.multiply(model['po_%i' %i], tf.nn.tanh(model['pz_%i' %i]), name = 'ph_%i' %i)

			model['ph_%i' %(-1)] = tf.zeros([dim_b, dim_i], tf.float32) #can remove
			model['ph_%i' %(dim_t)] = tf.zeros([dim_b, dim_i], tf.float32) #can remove

		#with tf.name_scope('output'):
		#	#to edit
		#	model['ph'] = tf.where(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['pFh_%i_%i' %(dim_d - 1, ii - 1)], model['pFh_%i_%i' %(ii)], name = 'ph')
		#	#to edit

		with tf.name_scope('meansquared'):
			for i in xrange(dim_t):
				model['pms_%i' %i] = tf.where(tf.equal(model['pxi_%i' %i], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.subtract(model['py_%i' %i], model['ph_%i' %i])), [1]), name = 'pms_%i' %i)
			model['pms'] = tf.reduce_sum(tf.add_n([model['pms_%i' %i] for i in xrange(dim_t)]), name = 'pms')
			model['sp+ms'] = tf.summary.scalar(model['pms'].name, model['pms'])

	#insert hypothesis lstm  here

	with tf.name_scope('hlstm'):
		with tf.name_scope('input'):
			for i in xrange(dim_t):
				model['hxi_%i' %i] = tf.placeholder(tf.int32, [dim_b], name = 'hxi_%i' %i)
				model['hx_%i' %i] = tf.add(tf.nn.embedding_lookup(model['We'], model['hxi_%i' %i]), model['Be'], name = 'hx_%i' %i)

		with tf.name_scope('label'):
			for i in xrange(dim_t):
				model['hyi_%i' %i] = tf.placeholder(tf.int32, [dim_b], name = 'hyi_%i' %i)
				model['hy_%i' %i] = tf.add(tf.nn.embedding_lookup(model['We'], model['hyi_%i' %i]), model['Be'], name = 'hy_%i' %i)

		#with tf.name_scope('inputs'):
		#	for i in xrange(dim_t):
		#		model['hFx_%i' %i] = model['hx_%i' %i]
		#		model['hBx_%i' %i] = model['hx_%i' %i]

		# build the 3 gates i/p forget and o/p
		with tf.name_scope('inputgate'):
			model['hWi'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWi')
			model['hBi'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBi')
			for i in xrange(dim_t):
				model['hi_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['hx_%i' %i], model['hWi']), model['hBi']), name = 'hi_%i' %i)

		with tf.name_scope('forgetgate'):
			model['hWf'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWf')
			model['hBf'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBf')
			for i in xrange(dim_t):
				model['hf_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['hx_%i' %i], model['hWf']), model['hBf']), name = 'hf_%i' %i)

		with tf.name_scope('outputgate'):
			model['hWo'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWo')
			model['hBo'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBo')
			for i in xrange(dim_t):
				model['ho_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['hx_%i' %i], model['hWo']), model['hBo']), name = 'ho_%i' %i)

		with tf.name_scope('cellstate'):
			model['hWc'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWc')
			model['hBc'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBc')
			for i in xrange(dim_t):
				model['hcc_%i' %i] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'hcc_%i' %i) # consider starting with all zeros

			for i in xrange(dim_t):
				if i % 4 == 0 or i % 4 == 3:
					model['hc_%i' %i] = tf.where(tf.equal(model['hxi_%i' %i], tf.zeros([dim_b], tf.int32)), model['hcc_%i' %i], tf.multiply(model['hi_%i' %i], tf.nn.tanh(tf.add(tf.matmul(model['hx_%i' %i], model['hWc']), model['hBc']))), name = 'hc_%i' %i) #consider the 2 children input states
			for i in xrange(dim_t):
				if i % 4 == 1:
					model['hc_%i' %i] = tf.where(tf.equal(model['hxi_%i' %i], tf.zeros([dim_b], tf.int32)), model['hcc_%i' %i], tf.add(tf.to_float(tf.add(tf.multiply(model['hf_%i' %(i-1)], model['hc_%i' %(i-1)]), tf.multiply(model['hf_%i' %(i+2)], model['hc_%i' %(i+2)]))), tf.multiply(model['hi_%i' %i], tf.nn.tanh(tf.add(tf.matmul(model['hx_%i' %i], model['hWc']), model['hBc'])))), name = 'hc_%i' %i) #consider the 2 children input states
			for i in xrange(dim_t):
				if i % 4 == 2:
					model['hc_%i' %i] = tf.where(tf.equal(model['hxi_%i' %i], tf.zeros([dim_b], tf.int32)), model['hcc_%i' %i], tf.add(tf.to_float(tf.add(tf.multiply(model['hf_%i' %(i-1)], model['hc_%i' %(i-1)]), tf.multiply(model['hf_%i' %((i+3)%dim_t)], model['hc_%i' %((i+3)%dim_t)]))), tf.multiply(model['hi_%i' %i], tf.nn.tanh(tf.add(tf.matmul(model['hx_%i' %i], model['hWc']), model['hBc'])))), name = 'hc_%i' %i) #consider the 2 children input states

		with tf.name_scope('hidden_%i' %i):
			model['hWz'] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWz')
			model['hBz'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBz')
			for i in xrange(dim_t):
				model['hz_%i' %i] = tf.add(tf.matmul(model['hc_%i' %i], model['hWz']), model['hBz'], name = 'hz_%i' %i)

		with tf.name_scope('outputs'):
			for i in xrange(dim_t):
				model['hh_%i' %i] = tf.multiply(model['ho_%i' %i], tf.nn.tanh(model['hz_%i' %i]), name = 'hh_%i' %i)

			model['hh_%i' %(-1)] = tf.zeros([dim_b, dim_i], tf.float32) #can remove
			model['hh_%i' %(dim_t)] = tf.zeros([dim_b, dim_i], tf.float32) #can remove

		#with tf.name_scope('output'):
		#	#to edit
		#	model['ph'] = tf.where(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['pFh_%i_%i' %(dim_d - 1, ii - 1)], model['pFh_%i_%i' %(ii)], name = 'ph')
		#	#to edit

		with tf.name_scope('meansquared'):
			for i in xrange(dim_t):
				model['hms_%i' %i] = tf.where(tf.equal(model['hxi_%i' %i], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.subtract(model['hy_%i' %i], model['hh_%i' %i])), [1]), name = 'hms_%i' %i)
			model['hms'] = tf.reduce_sum(tf.add_n([model['hms_%i' %i] for i in xrange(dim_t)]), name = 'hms')
			model['sh+ms'] = tf.summary.scalar(model['hms'].name, model['hms'])

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
	model['tms'] = optim_ms(model['lrms']).minimize(model['pms'] + model['hms'], global_step = model['gsms'], name = 'tms')

	model['gsce'] = tf.Variable(0, trainable = False, name = 'gsce')
	model['lrce'] = tf.train.exponential_decay(lrate_ce, model['gsce'], dstep_ce, drate_ce, staircase = False, name = 'lrce')
	model['tce'] = optim_ce(model['lrce']).minimize(model['cce'], global_step = model['gsce'], name = 'tce')

	return model			

