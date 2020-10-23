Discriminator(
  (rgb_to_features): ModuleList(
    (0): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
    (1): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
    (2): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
    (3): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
    (4): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1))
    (5): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1))
  )
  (final_converter): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
  (layers): ModuleList(
    (0): DisGeneralConvBlock(
      (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (downSampler): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (1): DisGeneralConvBlock(
      (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (downSampler): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (2): DisGeneralConvBlock(
      (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (downSampler): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (3): DisGeneralConvBlock(
      (conv_1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (downSampler): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (4): DisGeneralConvBlock(
      (conv_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (downSampler): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (5): DisGeneralConvBlock(
      (conv_1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (downSampler): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
  )
  (final_block): DisFinalBlock(
    (batch_discriminator): MinibatchStdDev()
    (conv_1): Conv2d(257, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv_2): Conv2d(256, 256, kernel_size=(4, 4), stride=(1, 1))
    (conv_3): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    (lrelu): LeakyReLU(negative_slope=0.2)
  )
)

Process finished with exit code 0
