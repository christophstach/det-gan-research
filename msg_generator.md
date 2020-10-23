Generator(
  (layers): ModuleList(
    (0): GenInitialBlock(
      (conv_1): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(1, 1))
      (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pixNorm): PixelwiseNorm()
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (1): GenGeneralConvBlock(
      (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pixNorm): PixelwiseNorm()
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (2): GenGeneralConvBlock(
      (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pixNorm): PixelwiseNorm()
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (3): GenGeneralConvBlock(
      (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pixNorm): PixelwiseNorm()
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (4): GenGeneralConvBlock(
      (conv_1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pixNorm): PixelwiseNorm()
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
    (5): GenGeneralConvBlock(
      (conv_1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pixNorm): PixelwiseNorm()
      (lrelu): LeakyReLU(negative_slope=0.2)
    )
  )
  (rgb_converters): ModuleList(
    (0): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    (1): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    (2): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    (3): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    (4): Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1))
    (5): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
  )
)
torch.Size([1, 3, 128, 128])

Process finished with exit code 0
