from ..basics.BaseClasses import ParameterizedLayer
import numpy as np

class Convolution(ParameterizedLayer):
    def __init__(self,channel_in:int,channel_out:int,filter_height:int,filter_width:int,rng: np.random.Generator |None =None,initializer:str="xavier")->None:
        super().__init__()
        if channel_in<=0 or channel_out<=0:
            raise ValueError("Convolution channel and channel out must be positive")
        if filter_height <= 0 or filter_width <= 0:
            raise ValueError("filter dimensions must be positive")
        self.generator= rng if rng is not None else np.random.default_rng()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.filter_height = filter_height
        self.filter_width = filter_width
        #以下为前向传播时才会产生的属性，用于维护反向传播
        self.input_shape = None
        self.padded_input = None
        self.padding = None
        self.stride = None
        self.output_shape =None
        fan_in = (
                channel_in
                * filter_height
                * filter_width
        )

        fan_out = (
                channel_out
                * filter_height
                * filter_width
        )

        scales = {
            "xavier": np.sqrt(2.0 / (fan_in + fan_out)),
            "he": np.sqrt(2.0 / fan_in),
            "small": 0.1,
        }
        if not (initializer in scales):
            raise ValueError(f"unknown initializer: {initializer!r}")
        self.filter=self.register_parameter(self.generator.standard_normal((channel_out,
                                                                            channel_in,
                                                                            filter_height,
                                                                            filter_width))
                                                                            *scales[initializer])
        self.bias=self.register_parameter(np.zeros(channel_out))


    def _padding(self,input:np.ndarray,padding:int)->np.ndarray:
        padded_input = np.pad(
            input,
            pad_width=(
                (0, 0),  # batch 不填充
                (0, 0),  # channels 不填充
                (padding, padding),  # height
                (padding, padding),  # width
            ),
            mode="constant",
            constant_values=0,
        )
        return padded_input

    def forward(self,input:np.ndarray,padding:int =0,stride:int=1) -> np.ndarray:
        if input.ndim != 4:
            raise ValueError("input must be 4D in order of: BatchSize,Channel(in),Height.Width")
        if input.shape[1]!=self.channel_in:
            raise ValueError(f"channels of filter and input image must correspond to size {self.channel_in}")
        if padding<0 or stride<=0:
            raise ValueError("padding and stride must both be positive")
        batch_size=input.shape[0]
        input_height=input.shape[2]
        input_width=input.shape[3]
        padded_input=self._padding(input,padding)
        if (
                input_height+2*padding < self.filter_height
                or input_width+2*padding < self.filter_width
        ):
            raise ValueError(
                "filter cannot be larger than the padded input"
            )
        #convolution implementing with tricky for loop--waiting to be optimized
        out_height=(input_height+2*padding-self.filter_height)//stride+1
        out_width=(input_width+2*padding-self.filter_width)//stride+1
        output=np.zeros((batch_size,
                         self.channel_out,
                         out_height,
                         out_width)
                        ,dtype=float)

        for batch in range(batch_size):
            for c_out in range(self.channel_out):
                current_filter = self.filter.data[c_out]

                for out_h in range(out_height):
                    h_start = out_h * stride
                    h_end = h_start + self.filter_height

                    for out_w in range(out_width):
                        w_start = out_w * stride
                        w_end = w_start + self.filter_width

                        input_patch = padded_input[
                            batch,
                            :,
                            h_start:h_end,
                            w_start:w_end,
                        ]

                        output[batch, c_out, out_h, out_w] = (
                                np.sum(input_patch * current_filter)
                                + self.bias.data[c_out]
                        )
                #注意：bias只在所有channel_in求和完成后在所有相应位置加一次，而不是每个channel_in都加一次
                output[batch,c_out,:,:]+=self.bias.data[c_out]
        #保存这个变量用于给filter反向传播求梯度
        self.input_shape=input.shape
        self.padded_input=padded_input
        self.padding=padding
        self.stride=stride
        self.output_shape=output.shape
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not hasattr(self, "padded_input"):
            raise RuntimeError("forward must be called before backward")

        grad_output = np.asarray(grad_output, dtype=float)

        if grad_output.shape != self.output_shape:
            raise ValueError(
                f"expected grad_output shape {self.output_shape}, "
                f"got {grad_output.shape}"
            )

        batch_size, _, out_height, out_width = grad_output.shape

        grad_padded_input = np.zeros_like(self.padded_input)
        grad_filter = np.zeros_like(self.filter.data)

        for batch in range(batch_size):
            for c_out in range(self.channel_out):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        h_start = out_h * self.stride
                        w_start = out_w * self.stride

                        input_patch = self.padded_input[
                            batch,
                            :,
                            h_start: h_start + self.filter_height,
                            w_start: w_start + self.filter_width,
                        ]

                        gradient = grad_output[
                            batch,
                            c_out,
                            out_h,
                            out_w,
                        ]

                        # 当前输出位置对 filter 的贡献
                        grad_filter[c_out] += gradient * input_patch

                        # 当前输出位置对 input patch 的贡献
                        grad_padded_input[
                            batch,
                            :,
                            h_start: h_start + self.filter_height,
                            w_start: w_start + self.filter_width,
                        ] += gradient * self.filter.data[c_out]

        self.filter.grad[...] = grad_filter

        self.bias.grad[...] = np.sum(
            grad_output,
            axis=(0, 2, 3),
        )

        if self.padding == 0:
            return grad_padded_input

        return grad_padded_input[
            :,
            :,
            self.padding: -self.padding,
            self.padding: -self.padding,
        ]
