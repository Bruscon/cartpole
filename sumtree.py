
# There are two SumTree classes here: one is pure python adapted from:
# https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py

# The other is fully vectorized and runs on GPU. It is MUCH FASTER for large arrays!

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



import tensorflow as tf

class TFSumTree:
    def __init__(self, size):
        self.size = size
        self.nodes = tf.Variable(tf.zeros(2 * size - 1, dtype=tf.float32))
        self.priorities = tf.Variable(tf.zeros(size, dtype=tf.float32))  # Store priorities separately
        self.data = tf.Variable(tf.zeros(size, dtype=tf.float32))  # Store actual data
        self.count = tf.Variable(0, dtype=tf.int32)
        self.real_size = tf.Variable(0, dtype=tf.int32)
    
    @property
    def total(self):
        return self.nodes[0]
    
    def rebuild_tree(self):
        """Rebuild entire tree from leaf nodes using NumPy"""
        # Get numpy arrays
        nodes_np = self.nodes.numpy()
        priorities_np = self.priorities.numpy()  # Use priorities, not data!
        
        # Copy priorities to leaf positions in tree
        # In a complete binary tree, leaves start at index size-1
        nodes_np[self.size - 1:2 * self.size - 1] = priorities_np[:self.size]
        
        # Build internal nodes bottom-up
        for i in range(self.size - 2, -1, -1):
            nodes_np[i] = nodes_np[2 * i + 1] + nodes_np[2 * i + 2]
        
        # Assign back to TF variable once
        self.nodes.assign(nodes_np)
    
    @tf.function
    def vectorized_add_leaves(self, values, data_values):
        """Add multiple items to leaves only - no tree update"""
        batch_size = tf.shape(values)[0]
        start_count = self.count
        
        # Calculate indices for all items
        indices = tf.range(batch_size) + start_count
        indices = tf.math.mod(indices, self.size)
        
        # Update priorities array
        self.priorities.assign(tf.tensor_scatter_nd_update(
            self.priorities,
            tf.expand_dims(indices, 1),
            values  # Store priorities
        ))
        
        # Update data array
        self.data.assign(tf.tensor_scatter_nd_update(
            self.data,
            tf.expand_dims(indices, 1),
            data_values  # Store actual data
        ))
        
        # Update count and real_size
        new_count = tf.math.mod(start_count + batch_size, self.size)
        self.count.assign(new_count)
        self.real_size.assign(tf.minimum(self.size, self.real_size + batch_size))
    
    @tf.function
    def vectorized_get(self, cumsums):
        """Vectorized get for multiple queries"""
        batch_size = tf.shape(cumsums)[0]
        
        # Initialize
        indices = tf.zeros(batch_size, dtype=tf.int32)
        remaining = tf.identity(cumsums)
        
        # Maximum depth of tree
        max_depth = tf.cast(tf.math.ceil(tf.math.log(tf.cast(self.size * 2 - 1, tf.float32)) / tf.math.log(2.0)), tf.int32)
        
        # Traverse tree for all queries simultaneously
        for _ in range(max_depth):
            left_indices = 2 * indices + 1
            right_indices = 2 * indices + 2
            
            # Bounds check
            valid = left_indices < tf.shape(self.nodes)[0]
            
            # Get left values
            left_vals = tf.where(valid, tf.gather(self.nodes, left_indices), 0.0)
            
            # Decide direction for each query
            go_left = remaining < left_vals
            go_left = tf.logical_and(go_left, valid)
            
            # Update indices
            indices = tf.where(go_left, left_indices, right_indices)
            indices = tf.where(valid, indices, indices)  # Keep same if invalid
            
            # Update remaining cumsum
            remaining = tf.where(go_left, remaining, remaining - left_vals)
        
        # Convert to data indices
        data_indices = indices - self.size + 1
        
        # Gather results
        priorities = tf.gather(self.nodes, indices)
        data_vals = tf.gather(self.data, data_indices)  # Return actual data, not priorities
        
        return data_indices, priorities, data_vals
    
    def batch_add(self, values, data_values):
        """Add multiple items - leaves only, no tree update"""
        values = tf.constant(values, dtype=tf.float32)
        data_values = tf.constant(data_values, dtype=tf.float32)
        self.vectorized_add_leaves(values, data_values)
    
    def batch_get(self, cumsums):
        """Get multiple items efficiently"""
        cumsums = tf.constant(cumsums, dtype=tf.float32)
        return self.vectorized_get(cumsums)



# Example usage
if __name__ == "__main__":
	import numpy as np
	
	# Tree sizes to test
	SIZE = 500_000
	SAMPLES = 5_000
	
	print(f"Creating trees with {SIZE} nodes...")
	
	# Original Python SumTree
	py_tree = SumTree(SIZE)
	
	# TensorFlow SumTree
	tf_tree = TFSumTree(SIZE)
	
	# Fill both trees
	print("Filling trees...")
	np.random.seed(42)
	priorities = np.random.uniform(0.1, 10.0, SIZE).astype(np.float32)
	data_values = np.arange(SIZE, dtype=np.float32)
	
	# Python tree - sequential
	for i in range(SIZE):
		if i % (SIZE/10) == 0:
			print(f"  Added {i}/{SIZE} nodes...")
		py_tree.add(priorities[i], float(i))
	
	# TensorFlow tree - batch add
	batch_size = 1000
	for i in range(0, SIZE, batch_size):
		if i % (SIZE/10) == 0:
			print(f"  Added {i}/{SIZE} nodes...")
		end = min(i + batch_size, SIZE)
		tf_tree.batch_add(priorities[i:end], data_values[i:end])

	# Rebuild tree once after all additions
	print("Rebuilding TF tree...")
	tf_tree.rebuild_tree()
	
	print(f"\nTotal sum - Python: {py_tree.total}, TensorFlow: {tf_tree.total.numpy()}")
	
	# Generate random sample points
	sample_points = np.random.uniform(0, py_tree.total * 0.99, SAMPLES).astype(np.float32)
	
	# Python version timing
	def py_test():
		for s in sample_points:
			py_tree.get(s)
	
	# TensorFlow batch timing
	@tf.function
	def tf_batch_test():
		return tf_tree.batch_get(sample_points)
	
	# Warmup
	print("\nWarming up TensorFlow...")
	_ = tf_batch_test()
	
	# Time both versions
	print(f"\nTiming {SAMPLES} get() operations:")
	
	py_time = timeit.timeit(py_test, number=1)
	print(f"Python SumTree: {py_time:.4f} seconds")
	
	tf_time = timeit.timeit(tf_batch_test, number=1)
	print(f"TensorFlow SumTree (batch): {tf_time:.4f} seconds")
	
	print(f"\nSpeedup: {py_time/tf_time:.2f}x")
	
	# Verify correctness
	print("\nVerifying correctness (first 3 samples):")
	tf_results = tf_tree.batch_get(sample_points[:3])
	for i in range(3):
		py_idx, py_priority, py_data = py_tree.get(sample_points[i])
		tf_idx = tf_results[0][i].numpy()
		print(f"  Sample {sample_points[i]:.2f}: Python idx={py_idx}, TF idx={tf_idx}")

