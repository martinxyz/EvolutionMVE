import json
import numpy as np
import scipy.misc
from tensorflow import keras
from evolve import Catch
model_from_json = keras.models.model_from_json


if __name__ == "__main__":
    # Make sure this grid size matches the value used fro training
    grid_size = 10

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    for e in range(30):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        img = input_t.reshape((grid_size,)*2) * 255
        for n in range(5):
            scipy.misc.imsave("%04d.png" % c, img)
            c += 1
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            img = input_t.reshape((grid_size,)*2)
            for n in range(5):
                scipy.misc.imsave("%04d.png" % c, img)
                c += 1
