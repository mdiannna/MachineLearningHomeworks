from data_generator import training_generator, test_generator

generator = training_generator(batch_size=8) # batch size of 8

x, numbers, numbers_sum = next(generator)

# x.shape == (8, 2, 28, 84)     # 8 pairs of images with height 28px and width 84px
# numbers.shape == (8, 2)       # 8 pairs of numbers corresponding to the images
# numbers_sum.shape == (8, 1)   # 8 numbers that represent the sum of the numbers from the images
