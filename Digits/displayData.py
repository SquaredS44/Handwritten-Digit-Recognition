import math
import matplotlib.pyplot as plt
import numpy as np


def displayData(x, example_width=None):
    # DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the
    #   displayed array if requested.

    # closes previously opened figure. preventing a
    # warning after opening too many figures
    plt.close()

    # creates new figure
    plt.figure()

    # turns 1D X array into 2D
    if x.ndim == 1:
        x = np.reshape(x, (-1, x.shape[0]))

    # set example_width automatically if not passed in
    if not example_width or not 'example_width' in locals():
        example_width = int(round(math.sqrt(x.shape[1])))

    # gray image
    plt.set_cmap("gray")

    # compute rows, cols
    m, n = x.shape
    example_height = int(n / example_width)

    # compute number of items to display
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))

    # between images padding
    pad = 1

    # setup blank display
    a= pad+display_rows*(example_height+pad)
    b = pad+display_cols*(example_width+pad)
    display_array = -np.ones((a,b))

    # copy each example into a patch on the display array
    curr_ex = 1
    for j in range(1, display_rows + 1):
        for i in range(1, display_cols + 1):
            if curr_ex > m:
                break

            # copy the patch

            # get the max value of the patch to normalize all examples
            max_val = max(abs(x[curr_ex - 1, :]))
            print(example_height)
            print(example_width)
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width + pad) + np.array(range(example_width))
            display_array[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = np.reshape(x[curr_ex - 1, :],
                                                                                   (example_height, example_width),
                                                                                   order="F") / max_val
            curr_ex += 1

        if curr_ex > m:
            break

    # display image
    h = plt.imshow(display_array, vmin=-1, vmax=1)

    # do not show axis
    plt.axis('off')

    plt.show(block=False)

    return h, display_array