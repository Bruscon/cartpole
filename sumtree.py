# There are three SumTree classes here: one is pure python adapted from:
# https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py

# The second is vectorized with numpy and runs marginally faster (<2x)

# The other is fully vectorized and runs on GPU. It samples much faster (>5x) for large arrays

# There is one catch. Adding elements to the tree only updates leaf nodes. The rest of the tree
# needs to be updated by calling rebuild_tree for the sums to be valid.

# This is because updating the tree cannot be vectorized. It is still much faster this way!

# Usage examples in __main__ at the bottom


import timeit

class SumTree:
	def __init__(self, size):
		self.nodes = [0] * (2 * size - 1)
		self.data = [None] * size

		self.size = size
		self.count = 0
		self.real_size = 0

	@property
	def total(self):
		return self.nodes[0]

	def update(self, data_idx, value):
		idx = data_idx + self.size - 1  # child index in tree array
		change = value - self.nodes[idx]

		self.nodes[idx] = value

		parent = (idx - 1) // 2
		while parent >= 0:
			self.nodes[parent] += change
			parent = (parent - 1) // 2

	def add(self, value, data):
		self.data[self.count] = data
		self.update(self.count, value)

		self.count = (self.count + 1) % self.size #modulo makes this a cicular buffer
		self.real_size = min(self.size, self.real_size + 1)

	def get(self, cumsum):
		assert cumsum <= self.total
		idx = 0
		while 2 * idx + 1 < len(self.nodes):
			left, right = 2*idx + 1, 2*idx + 2
			
			if right >= len(self.nodes) or cumsum < self.nodes[left]:
				idx = left
			else:
				cumsum -= self.nodes[left]
				idx = right
				
		data_idx = idx - self.size + 1
		return data_idx, self.nodes[idx], self.data[data_idx]

	def __repr__(self):
		return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


import numpy as np

class NPSumTree:
	def __init__(self, size):
		self.size = size
		self.tree_size = 2 * size - 1
		
		# Use float32 for nodes to save memory while maintaining precision
		self.nodes = np.zeros(self.tree_size, dtype=np.float32)
		
		# Store data as object array to handle any data type
		self.data = np.empty(size, dtype=object)
		
		self.count = 0
		self.real_size = 0
		
		# Pre-compute parent indices for faster updates
		self.parent_indices = np.zeros(self.tree_size, dtype=np.int32)
		for i in range(1, self.tree_size):
			self.parent_indices[i] = (i - 1) // 2
			
	@property
	def total(self):
		return self.nodes[0]
	
	def update(self, data_idx, value):
		"""Update a single value and propagate changes up the tree immediately"""
		# Convert to tree index
		idx = data_idx + self.size - 1
		
		# Calculate change
		change = value - self.nodes[idx]
		if change == 0:
			return
			
		# Update leaf
		self.nodes[idx] = value
		
		# Propagate change up the tree
		parent = self.parent_indices[idx]
		while parent >= 0:
			self.nodes[parent] += change
			if parent == 0:
				break
			parent = self.parent_indices[parent]
	
	def batch_update(self, data_indices, values):
		"""Update multiple values at once and propagate changes up the tree immediately"""
		data_indices = np.asarray(data_indices, dtype=np.int32)
		values = np.asarray(values, dtype=np.float32)
		
		# Convert to tree indices
		tree_indices = data_indices + self.size - 1
		
		# Calculate changes
		changes = values - self.nodes[tree_indices]
		
		# Update leaves
		self.nodes[tree_indices] = values
		
		# Propagate changes up the tree
		# Group by parent to minimize redundant updates
		for idx, change in zip(tree_indices, changes):
			if change == 0:
				continue
			parent = self.parent_indices[idx]
			while parent >= 0:
				self.nodes[parent] += change
				if parent == 0:
					break
				parent = self.parent_indices[parent]
	
	def add(self, value, data):
		self.data[self.count] = data
		# Directly set the leaf value without updating parents
		leaf_idx = self.count + self.size - 1
		self.nodes[leaf_idx] = value
		self.count = (self.count + 1) % self.size
		self.real_size = min(self.size, self.real_size + 1)
	
	def batch_add(self, values, data_list):
		"""Add multiple items at once"""
		n = len(values)
		if n == 0:
			return
			
		# Calculate indices for new items
		indices = np.arange(self.count, self.count + n) % self.size
		
		# Store data
		for i, idx in enumerate(indices):
			self.data[idx] = data_list[i]
			
		# Directly set leaf values without updating parents
		leaf_indices = indices + self.size - 1
		self.nodes[leaf_indices] = values
		
		# Update count and real_size
		self.count = (self.count + n) % self.size
		self.real_size = min(self.size, self.real_size + n)
	
	def rebuild_tree(self):
		"""Efficiently rebuild all internal node sums from leaf values"""
		# Work from the last internal node up to the root
		for i in range(self.size - 2, -1, -1):
			left = 2 * i + 1
			right = 2 * i + 2
			
			# Sum the children
			self.nodes[i] = self.nodes[left]
			if right < self.tree_size:
				self.nodes[i] += self.nodes[right]
	
	def get(self, cumsum):
		"""Sample based on cumulative sum. Note: rebuild_tree() must be called after add operations"""
		assert cumsum <= self.total, f"cumsum {cumsum} exceeds total {self.total}"
		
		idx = 0
		cumsum = np.float32(cumsum)  # Ensure same dtype
		
		# Traverse down the tree
		while 2 * idx + 1 < self.tree_size:
			left = 2 * idx + 1
			right = 2 * idx + 2
			
			left_sum = self.nodes[left]
			
			if cumsum < left_sum:
				idx = left
			else:
				cumsum -= left_sum
				idx = right
				
		data_idx = idx - self.size + 1
		return data_idx, self.nodes[idx], self.data[data_idx]
	
	def batch_get(self, cumsums):
		"""Get multiple samples at once - vectorized where possible"""
		cumsums = np.asarray(cumsums, dtype=np.float32)
		n = len(cumsums)
		
		data_indices = np.zeros(n, dtype=np.int32)
		values = np.zeros(n, dtype=np.float32)
		data_items = []
		
		for i, cumsum in enumerate(cumsums):
			data_idx, value, data = self.get(cumsum)
			data_indices[i] = data_idx
			values[i] = value
			data_items.append(data)
			
		return data_indices, values, data_items
	
	def get_all_values(self):
		"""Get all current values as a numpy array"""
		leaf_start = self.size - 1
		return self.nodes[leaf_start:leaf_start + self.real_size].copy()
	
	def __repr__(self):
		return f"SumTree(size={self.size}, real_size={self.real_size}, total={self.total:.6f})"
	
	def __len__(self):
		return self.real_size


# Example usage and performance test
if __name__ == "__main__":
	import time
	
	# Create tree
	size = 500_000
	tree = NPSumTree(size)
	
	# Test single adds
	print("Testing single adds...")
	start = time.time()
	for i in range(size):
		tree.add(np.random.random(), f"data_{i}")
	tree.rebuild_tree()  # Rebuild once after all adds
	single_time = time.time() - start
	print(f"Single adds with rebuild: {single_time:.4f}s")
	
	# Test batch adds
	tree2 = NPSumTree(size)
	print("\nTesting batch adds...")
	start = time.time()
	batch_size = 1000
	for i in range(0, size, batch_size):
		values = np.random.random(batch_size)
		data = [f"data_{j}" for j in range(i, i + batch_size)]
		tree2.batch_add(values, data)
	tree2.rebuild_tree()  # Rebuild once after all adds
	batch_time = time.time() - start
	print(f"Batch adds with rebuild: {batch_time:.4f}s")
	print(f"Speedup: {single_time/batch_time:.2f}x")
	
	# Test sampling
	print("\nTesting sampling...")
	n_samples = 5000
	start = time.time()
	for _ in range(n_samples):
		cumsum = np.random.random() * tree.total
		idx, value, data = tree.get(cumsum)
	sample_time = time.time() - start
	print(f"Single samples: {sample_time:.4f}s ({n_samples/sample_time:.0f} samples/sec)")
	
	# Test batch sampling
	start = time.time()
	cumsums = np.random.random(n_samples) * tree.total
	indices, values, data_items = tree.batch_get(cumsums)
	batch_sample_time = time.time() - start
	print(f"Batch samples: {batch_sample_time:.4f}s ({n_samples/batch_sample_time:.0f} samples/sec)")
	print(f"Sampling speedup: {sample_time/batch_sample_time:.2f}x")


import tensorflow as tf

class TFSumTree:
   def __init__(self, size):
       self.size = size
       # Numpy storage for fast additions and rebuilds
       self.nodes = np.zeros(2 * size - 1, dtype=np.float32)
       self.priorities = np.zeros(size, dtype=np.float32)
       self.data = np.zeros(size, dtype=np.float32)
       self.count = 0
       self.real_size = 0
       
       # TensorFlow variables for sampling (created on-demand)
       self._tf_nodes = None
       self._tf_data = None
   
   @property
   def total(self):
       return self.nodes[0]
   
   def batch_add(self, values, data_values):
       """Add multiple items sequentially using numpy"""
       values = np.asarray(values, dtype=np.float32)
       data_values = np.asarray(data_values, dtype=np.float32)
       batch_size = len(values)
       
       # Sequential assignment (no wraparound since size is multiple of batch size)
       start_idx = self.count
       end_idx = start_idx + batch_size
       
       # Update arrays
       self.priorities[start_idx:end_idx] = values
       self.data[start_idx:end_idx] = data_values
       
       # Update counters
       self.count = (self.count + batch_size) % self.size
       self.real_size = min(self.size, self.real_size + batch_size)
   
   def rebuild_tree(self, start_leaf=None, end_leaf=None):
       """Rebuild tree using numpy for maximum speed
       
       Args:
           start_leaf: Starting leaf index (inclusive). If None, rebuilds entire tree.
           end_leaf: Ending leaf index (exclusive). If None, rebuilds entire tree.
       """
       if start_leaf is None or end_leaf is None:
           # Full rebuild - copy all priorities to leaf positions
           self.nodes[self.size - 1:2 * self.size - 1] = self.priorities
           
           # Build internal nodes bottom-up
           for i in range(self.size - 2, -1, -1):
               self.nodes[i] = self.nodes[2 * i + 1] + self.nodes[2 * i + 2]
       else:
           # Partial rebuild for specified leaf range
           leaf_start = self.size - 1 + start_leaf
           leaf_end = self.size - 1 + end_leaf
           
           # Update affected leaf nodes
           self.nodes[leaf_start:leaf_end] = self.priorities[start_leaf:end_leaf]
           
           # Find and update all affected internal nodes
           affected_internal = set()
           for leaf_idx in range(leaf_start, leaf_end):
               current = leaf_idx
               while current > 0:
                   parent = (current - 1) // 2
                   affected_internal.add(parent)
                   current = parent
           
           # Update affected internal nodes bottom-up
           for i in sorted(affected_internal, reverse=True):
               self.nodes[i] = self.nodes[2 * i + 1] + self.nodes[2 * i + 2]
   
   def _sync_to_tensorflow(self):
       """Copy numpy arrays to TensorFlow tensors for sampling"""
       # Create or update TensorFlow variables
       if self._tf_nodes is None:
           self._tf_nodes = tf.Variable(self.nodes, dtype=tf.float32)
           self._tf_data = tf.Variable(self.data, dtype=tf.float32)
       else:
           self._tf_nodes.assign(self.nodes)
           self._tf_data.assign(self.data)
   
   @tf.function
   def _vectorized_get(self, cumsums, tf_nodes, tf_data):
       """Vectorized get using TensorFlow for maximum sampling speed"""
       batch_size = tf.shape(cumsums)[0]
       
       # Initialize at root
       indices = tf.zeros(batch_size, dtype=tf.int32)
       remaining = tf.identity(cumsums)
       
       # Tree properties
       tree_size = tf.shape(tf_nodes)[0]
       max_depth = tf.cast(tf.math.ceil(tf.math.log(tf.cast(tree_size, tf.float32)) / tf.math.log(2.0)), tf.int32)
       
       # Traverse tree
       for _ in tf.range(max_depth):
           left_children = 2 * indices + 1
           right_children = 2 * indices + 2
           
           # Check if we've reached leaves
           is_leaf = left_children >= tree_size
           
           # Get left child values (safe indexing)
           safe_left_children = tf.where(is_leaf, 0, left_children)
           left_values = tf.gather(tf_nodes, safe_left_children)
           left_values = tf.where(is_leaf, 0.0, left_values)
           
           # Decide direction
           go_left = remaining < left_values
           
           # Update indices for non-leaf nodes
           new_indices = tf.where(go_left, left_children, right_children)
           indices = tf.where(is_leaf, indices, new_indices)
           
           # Update remaining sum when going right
           should_subtract = tf.logical_and(tf.logical_not(go_left), tf.logical_not(is_leaf))
           remaining = tf.where(should_subtract, remaining - left_values, remaining)
       
       # Convert tree indices to data indices
       data_indices = indices - self.size + 1
       data_indices = tf.maximum(0, tf.minimum(data_indices, self.size - 1))
       
       # Gather results
       priorities = tf.gather(tf_nodes, indices)
       data_vals = tf.gather(tf_data, data_indices)
       
       return data_indices, priorities, data_vals
   
   def batch_get(self, cumsums):
       """Get multiple items efficiently using TensorFlow backend"""
       # Sync numpy data to TensorFlow
       self._sync_to_tensorflow()
       
       # Convert to tensor and clamp
       cumsums = tf.constant(cumsums, dtype=tf.float32)
       total = self.total
       safe_cumsums = tf.minimum(cumsums, total - 1e-7)
       safe_cumsums = tf.maximum(safe_cumsums, 0.0)
       
       # Use TensorFlow for vectorized sampling
       return self._vectorized_get(safe_cumsums, self._tf_nodes, self._tf_data)
   
   def single_add(self, priority, data_value):
       """Add a single item (for compatibility)"""
       self.batch_add([priority], [data_value])
   
   def single_get(self, cumsum):
       """Get a single item (for compatibility)"""
       result = self.batch_get([cumsum])
       return result[0][0].numpy(), result[1][0].numpy(), result[2][0].numpy()
   
   def __repr__(self):
       """String representation of the tree"""
       if self.real_size > 0:
           active_priorities = self.priorities[:self.real_size]
           min_p = np.min(active_priorities)
           max_p = np.max(active_priorities)
           return (f"HybridSumTree(size={self.size}, "
                   f"count={self.count}, "
                   f"real_size={self.real_size}, "
                   f"total={self.total:.2f}, "
                   f"min_priority={min_p:.4f}, "
                   f"max_priority={max_p:.4f})")
       else:
           return (f"HybridSumTree(size={self.size}, "
                   f"count={self.count}, "
                   f"real_size={self.real_size}, "
                   f"total={self.total:.2f}, "
                   f"empty=True)")

# Example usage
if __name__ == "__main__":	
    # Tree sizes to test
    SIZE            = 524_288      # 2^19
    SAMPLES         = 4_096        # 2^12
    BATCH_ADD_SIZE  = 1024         # 2^10
    
    print("="*60)
    print(f"SUMTREE PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Tree Size:        {SIZE:,} nodes")
    print(f"Batch Size:       {BATCH_ADD_SIZE:,} additions per batch")
    print(f"Sample Size:      {SAMPLES:,} get operations")
    print("="*60)
    
    # Create trees
    print("\nCreating trees...")
    py_tree = SumTree(SIZE)
    tf_tree = TFSumTree(SIZE)
    
    # Generate data once
    np.random.seed(42)
    priorities = np.random.uniform(0.1, 10.0, SIZE).astype(np.float32)
    data_values = np.arange(SIZE, dtype=np.float32)
    
    # Fill Python tree (sequential additions)
    print("\nFilling Python SumTree (sequential additions)...")
    py_start_time = time.time()
    
    for i in range(SIZE):
        if i % (SIZE//10) == 0 and i > 0:
            elapsed = time.time() - py_start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  Progress: {i:>7,}/{SIZE:,} ({i/SIZE*100:5.1f}%) - {rate:>8,.0f} adds/sec")
        py_tree.add(priorities[i], float(i))
    
    py_fill_time = time.time() - py_start_time
    print(f"  Python tree filled in {py_fill_time:.4f}s ({SIZE/py_fill_time:,.0f} adds/sec)")
    
    # Fill TensorFlow tree (batch additions)
    print("\nFilling TensorFlow SumTree (batch additions only)...")
    tf_add_start_time = time.time()
    total_rebuild_time = 0.0
    
    num_batches = SIZE // BATCH_ADD_SIZE
    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_ADD_SIZE
        batch_end = batch_start + BATCH_ADD_SIZE
        
        if batch_idx % (num_batches//10) == 0 and batch_idx > 0:
            elapsed = time.time() - tf_add_start_time
            rate = (batch_idx * BATCH_ADD_SIZE) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {batch_start:>7,}/{SIZE:,} ({batch_start/SIZE*100:5.1f}%) - {rate:>8,.0f} adds/sec")
        
        # Get current position before adding
        current_pos = tf_tree.count
        
        # Add batch to leaves (time only the addition)
        tf_tree.batch_add(priorities[batch_start:batch_end], data_values[batch_start:batch_end])
        
        # Time the rebuild separately
        rebuild_start = time.time()
        tf_tree.rebuild_tree(start_leaf=current_pos, end_leaf=current_pos + BATCH_ADD_SIZE)
        total_rebuild_time += time.time() - rebuild_start
    
    tf_add_time = time.time() - tf_add_start_time - total_rebuild_time
    print(f"  TensorFlow batch adds: {tf_add_time:.4f}s ({SIZE/tf_add_time:,.0f} adds/sec)")
    print(f"  Partial rebuilds:      {total_rebuild_time:.4f}s ({num_batches/total_rebuild_time:,.0f} rebuilds/sec)")
    
    # Test full tree rebuild
    print("\nTesting full tree rebuild...")
    full_rebuild_start = time.time()
    tf_tree.rebuild_tree()  # Full rebuild
    full_rebuild_time = time.time() - full_rebuild_start
    print(f"  Full rebuild:          {full_rebuild_time:.4f}s")
    
    # Compare totals
    print(f"\nTree Verification:")
    print(f"  Python total:     {py_tree.total:>12.2f}")
    print(f"  TensorFlow total: {tf_tree.total:>12.2f}")
    abs_diff = abs(py_tree.total - tf_tree.total)
    percent_diff = (abs_diff / py_tree.total) * 100 if py_tree.total > 0 else 0
    print(f"  Absolute diff:    {abs_diff:>12.6f}")
    print(f"  Percent diff:     {percent_diff:>12.6f}%")
    
    # Fill time comparison
    tf_total_time = tf_add_time + total_rebuild_time
    print(f"\nFill Time Breakdown:")
    print(f"  Python (sequential):      {py_fill_time:>8.4f}s")
    print(f"  TensorFlow (adds only):   {tf_add_time:>8.4f}s")
    print(f"  TensorFlow (rebuilds):    {total_rebuild_time:>8.4f}s")
    print(f"  TensorFlow (total):       {tf_total_time:>8.4f}s")
    print(f"  Full rebuild vs partial:  {full_rebuild_time:>8.4f}s vs {total_rebuild_time:>8.4f}s")
    if tf_total_time > 0:
        speedup = py_fill_time / tf_total_time
        print(f"  Overall speedup:          {speedup:>8.2f}x")
    
    # Generate sample points for get operations
    sample_points = np.random.uniform(0, py_tree.total * 0.99, SAMPLES).astype(np.float32)
    
    # Python get timing
    def py_test():
        for s in sample_points:
            py_tree.get(s)
    
    # TensorFlow batch get timing
    @tf.function
    def tf_batch_test():
        return tf_tree.batch_get(sample_points)
    
    # Warmup TensorFlow
    print(f"\nWarming up TensorFlow...")
    _ = tf_batch_test()
    
    # Time get operations
    print(f"\nTiming {SAMPLES:,} get() operations:")
    
    py_time = timeit.timeit(py_test, number=1)
    print(f"  Python (sequential):   {py_time:>8.4f}s ({SAMPLES/py_time:>8,.0f} gets/sec)")
    
    tf_time = timeit.timeit(tf_batch_test, number=1)
    print(f"  TensorFlow (batch):    {tf_time:>8.4f}s ({SAMPLES/tf_time:>8,.0f} gets/sec)")
    
    if tf_time > 0:
        get_speedup = py_time / tf_time
        print(f"  Get Speedup:           {get_speedup:>8.2f}x")
    
    # Verify correctness
    print(f"\nCorrectness Verification (first 3 samples):")
    tf_results = tf_tree.batch_get(sample_points[:3])
    all_correct = True
    
    for i in range(3):
        py_idx, py_priority, py_data = py_tree.get(sample_points[i])
        tf_idx = tf_results[0][i]
        tf_priority = tf_results[1][i]
        tf_data = tf_results[2][i]
        
        match = py_idx == tf_idx
        all_correct &= match
        status = "PASS" if match else "FAIL"
        
        print(f"  Sample {sample_points[i]:>8.2f}: Py_idx={py_idx:>6}, TF_idx={tf_idx:>6} {status}")
    
    print(f"\nFinal Results:")
    print(f"  Fill speedup:      {py_fill_time/tf_total_time:>6.2f}x")
    print(f"  Get speedup:       {py_time/tf_time:>6.2f}x") 
    print(f"  Rebuild efficiency:{total_rebuild_time/full_rebuild_time:>6.2f}x faster than full")
    print(f"  Correctness:       {'PASSED' if all_correct else 'FAILED'}")
    print("="*60)