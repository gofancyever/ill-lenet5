
from visdom import Visdom
import numpy as np
viz = Visdom()

Y = np.linspace(-5, 5, 100)
viz.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(
        markers=False,
        colormap='Electric',
              ),
)