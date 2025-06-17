from functools import reduce
import tensorboard

writer = tensorboard.SummaryWriter("./test")
for i in range(1000):
    # tf.summary.scalar(f'2*i', float(2*i), i)
    # tf.summary.scalar(f'5*i', float(5 * i), i)
    tf.summary.scalar(f'5*i', float(5 * i), i)
    writer.add_scalar(f'5*i', float(5 * i), i)
