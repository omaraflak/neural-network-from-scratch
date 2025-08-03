import losses
import modules
import trainer
import optimizers
import numpy as np

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2, 1))
y_train = np.array([[0], [1], [1], [0]]).reshape((4, 1, 1))

model = modules.Sequential([
    modules.Linear(2, 3),
    modules.Tanh(),
    modules.Linear(3, 1),
    modules.Tanh(),
])

trainer.train(
    model,
    x_train,
    y_train,
    losses.MSE(),
    optimizers.SGD(model, learning_rate=0.01),
    epochs=1000,
)

for x in x_train:
    print(x.tolist(), model.forward(x).tolist())
