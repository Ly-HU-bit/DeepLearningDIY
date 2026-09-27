from ..basics.BaseClasses import Layer
import numpy as np


class MaxPool(Layer):
    def __init__(
        self,
        pool_height: int,
        pool_width: int,
        pool_stride: int = 1,
        pool_padding: int = 0,
    ) -> None:
        super().__init__()

        if pool_padding < 0 or pool_stride <= 0:
            raise ValueError(
                "pool_padding must be non-negative "
                "and pool_stride must be positive"
            )

        if pool_height <= 0 or pool_width <= 0:
            raise ValueError(
                "pool_height and pool_width must be positive"
            )

        self.pool_height = pool_height
        self.pool_width = pool_width
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        # backward 所需缓存
        self.max_position = None
        self.pool_out_shape = None
        self.input_shape = None
        self.padded_input_shape = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=float)

        if input.ndim != 4:
            raise ValueError(
                "input must have shape "
                "(batch_size, channels, height, width), "
                f"but got {input.shape}"
            )

        batch_size, channels, input_height, input_width = input.shape

        padded_height = input_height + 2 * self.pool_padding
        padded_width = input_width + 2 * self.pool_padding

        if (
            padded_height < self.pool_height
            or padded_width < self.pool_width
        ):
            raise ValueError(
                "pool window cannot be larger than padded input"
            )

        # MaxPool padding 用 -inf，避免 padding 成为最大值
        padded_input = np.pad(
            input,
            (
                (0, 0),
                (0, 0),
                (self.pool_padding, self.pool_padding),
                (self.pool_padding, self.pool_padding),
            ),
            mode="constant",
            constant_values=-np.inf,
        )

        pool_out_height = (
            padded_height - self.pool_height
        ) // self.pool_stride + 1

        pool_out_width = (
            padded_width - self.pool_width
        ) // self.pool_stride + 1

        pool_output = np.empty(
            (
                batch_size,
                channels,
                pool_out_height,
                pool_out_width,
            ),
            dtype=float,
        )

        # 每一个 output 保存其 window 内最大值的位置
        self.max_position = np.empty(
            pool_output.shape,
            dtype=np.int64,
        )

        for batch in range(batch_size):
            for channel in range(channels):
                for out_h in range(pool_out_height):

                    h_start = out_h * self.pool_stride
                    h_end = h_start + self.pool_height

                    for out_w in range(pool_out_width):

                        w_start = out_w * self.pool_stride
                        w_end = w_start + self.pool_width

                        current_window = padded_input[
                            batch,
                            channel,
                            h_start:h_end,
                            w_start:w_end,
                        ]

                        max_index = int(current_window.argmax())

                        pool_output[
                            batch,
                            channel,
                            out_h,
                            out_w,
                        ] = current_window.flat[max_index]

                        self.max_position[
                            batch,
                            channel,
                            out_h,
                            out_w,
                        ] = max_index

        self.input_shape = input.shape
        self.padded_input_shape = padded_input.shape
        self.pool_out_shape = pool_output.shape

        return pool_output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.max_position is None:
            raise RuntimeError(
                "forward must be called before backward"
            )

        grad_output = np.asarray(grad_output, dtype=float)

        if grad_output.shape != self.pool_out_shape:
            raise ValueError(
                f"expected grad_output shape "
                f"{self.pool_out_shape}, "
                f"but got {grad_output.shape}"
            )

        (
            batch_size,
            channels,
            pool_out_height,
            pool_out_width,
        ) = self.pool_out_shape

        grad_padded_input = np.zeros(
            self.padded_input_shape,
            dtype=float,
        )

        for batch in range(batch_size):
            for channel in range(channels):
                for out_h in range(pool_out_height):

                    h_start = out_h * self.pool_stride

                    for out_w in range(pool_out_width):

                        w_start = out_w * self.pool_stride

                        max_index = self.max_position[
                            batch,
                            channel,
                            out_h,
                            out_w,
                        ]

                        max_h, max_w = np.unravel_index(
                            max_index,
                            (
                                self.pool_height,
                                self.pool_width,
                            ),
                        )

                        grad_padded_input[
                            batch,
                            channel,
                            h_start + max_h,
                            w_start + max_w,
                        ] += grad_output[
                            batch,
                            channel,
                            out_h,
                            out_w,
                        ]

        if self.pool_padding == 0:
            return grad_padded_input

        return grad_padded_input[
            :,
            :,
            self.pool_padding:-self.pool_padding,
            self.pool_padding:-self.pool_padding,
        ]


