"""
This is a code implementing a simple, single feature Linear Regression with Batch Gradient Descent from scratch using only vanilla Python
"""

class BatchLinearRegression:
	def __init__(self, learning_rate, num_epochs) -> None:
		self.weight = 0.0
		self.bias = 0.0
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs

	# Linear Regression functions
	def predict(self, x):
		return self.weight * x + self.bias  # y = wx + b, line formula
	
	def calculate_gradients(self, x, error):
		# Calculate the gradient using the derivative of the squared error loss function
		dldw = -1 * error * x  # dldw = -2 * xi * (y-ypred), but the -2 is constant meaning it can be abstracted, leaving only the negative sign and y-ypred = error
		dldb = -1 * error  # dldb = -2 * (y-ypred), but same thing as before

		return dldw, dldb

	def update_params(self, total_dldw, total_dldb, num_samples):
		# Calculate average gradients
		average_dldw = total_dldw / num_samples
		average_dldb = total_dldb / num_samples

		# Update parameters using Batch Gradient Descent
		self.weight = self.weight - self.learning_rate * average_dldw
		self.bias = self.bias - self.learning_rate * average_dldb

	def fit(self, inputs, outputs, print_epoch=True):
		for epoch in range(self.num_epochs):
			ase = 0.0
			total_dldw = 0.0
			total_dldb = 0.0

			for x, y in zip(inputs, outputs):
				# Make prediction
				ypred = self.predict(x)

				# Calculate error
				error = y-ypred

				# Calculate gradients
				current_dldw, current_dldb = self.calculate_gradients(x, error)

				# Sum up the calculated gradients to the total
				total_dldw += current_dldw
				total_dldb += current_dldb

				# Update the total error for the epoch
				ase += error ** 2

			num_samples = len(inputs)

			# Calculate MSE
			mse = ase / num_samples

			# Update Parameters
			self.update_params(total_dldw, total_dldb, num_samples)

			if print_epoch:
				# Print epoch results
				print(f"Epoch: {epoch} - Weight: {self.weight} - Bias: {self.bias} - MSE: {mse}")


# Creating the dataset
inputs = [1, 2, 3, 4, 5]
outputs = [26, 29, 32, 35, 38]  # 3x + 23, a randomly chosen linear function

# Creating the model
model = BatchLinearRegression(learning_rate=0.01, num_epochs=10000)

# Fitting the model to our dataset
model.fit(inputs, outputs)

# Print end results
print(f"\nFinal Results\nWeight: {model.weight} - Bias: {model.bias}")