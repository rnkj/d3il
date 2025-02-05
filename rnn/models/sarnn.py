import torch
import torch.nn as nn
from eipl.layer import InverseSpatialSoftmax, SpatialSoftmax
from IPython import embed as e


class SARNN(nn.Module):
    #:: SARNN
    """SARNN: Spatial Attention with Recurrent Neural Network.
    This model "explicitly" extracts positions from the image that are important to the task, such as the work object or arm position,
    and learns the time-series relationship between these positions and the robot's joint angles.
    The robot is able to generate robust motions in response to changes in object position and lighting.

    Arguments:
        rec_dim (int): The dimension of the recurrent state in the LSTM cell.
        k_dim (int, optional): The dimension of the attention points.
        joint_dim (int, optional): The dimension of the joint angles.
        temperature (float, optional): The temperature parameter for the softmax function.
        heatmap_size (float, optional): The size of the heatmap in the InverseSpatialSoftmax layer.
        kernel_size (int, optional): The size of the convolutional kernel.
        activation (str, optional): The name of activation function.
        im_size (list, optional): The size of the input image [height, width].
    """

    IMG_KEYS = ("bp", "inhand")

    def __init__(
        self,
        rec_dim,
        obs_dim,
        action_dim,
        k_dim=5,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        im_size=(96, 96),
    ):
        super(SARNN, self).__init__()

        self.k_dim = k_dim
        activation = nn.LeakyReLU(negative_slope=0.3)

        sub_im_size = [
            im_size[0] - 3 * (kernel_size - 1),
            im_size[1] - 3 * (kernel_size - 1),
        ]
        self.temperature = temperature
        self.heatmap_size = heatmap_size

        # Positional Encoder
        self.pos_encoders = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
                    activation,
                    nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
                    activation,
                    nn.Conv2d(
                        32, self.k_dim, 3, 1, 0
                    ),  # Convolutional layer 3
                    activation,
                    SpatialSoftmax(
                        width=sub_im_size[0],
                        height=sub_im_size[1],
                        temperature=self.temperature,
                        normalized=True,
                    ),  # Spatial Softmax layer
                )
                for key in self.IMG_KEYS
            }
        )
        # Image Encoder
        self.im_encoders = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
                    activation,
                    nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
                    activation,
                    nn.Conv2d(
                        32, self.k_dim, 3, 1, 0
                    ),  # Convolutional layer 3
                    activation,
                )
                for key in self.IMG_KEYS
            }
        )

        num_imgs = len(self.IMG_KEYS)
        rec_in = obs_dim + self.k_dim * 2 * num_imgs
        self.rec = nn.LSTMCell(rec_in, rec_dim)  # LSTM cell

        # Joint Decoder
        self.decoder_action = nn.Linear(rec_dim, action_dim)

        # Point Decoder
        self.decoder_point = nn.Sequential(
            nn.Linear(rec_dim, self.k_dim * 2 * num_imgs), activation
        )  # Linear layer and activation

        # Inverse Spatial Softmax
        self.issms = nn.ModuleDict(
            {
                key: InverseSpatialSoftmax(
                    width=sub_im_size[0],
                    height=sub_im_size[1],
                    heatmap_size=self.heatmap_size,
                    normalized=True,
                )
                for key in self.IMG_KEYS
            }
        )

        # Image Decoder
        self.decoder_images = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.ConvTranspose2d(
                        self.k_dim, 32, 3, 1, 0
                    ),  # Transposed Convolutional layer 1
                    activation,
                    nn.ConvTranspose2d(
                        32, 16, 3, 1, 0
                    ),  # Transposed Convolutional layer 2
                    activation,
                    nn.ConvTranspose2d(
                        16, 3, 3, 1, 0
                    ),  # Transposed Convolutional layer 3
                    activation,
                )
                for key in self.IMG_KEYS
            }
        )

        self.apply(self._weights_init)

    def _weights_init(self, m):
        """
        Tensorflow/Keras-like initialization
        """
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)

        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xi_dic, obs, rnn_state=None):
        """
        Forward pass of the SARNN module.
        Predicts the image, joint angle, and attention at the next time based on the image and joint angle at time t.
        Predict the image, joint angles, and attention points for the next state (t+1) based on
        the image and joint angles of the current state (t).
        By inputting the predicted joint angles as control commands for the robot,
        it is possible to generate sequential motion based on sensor information.
        """
        for key in self.IMG_KEYS:
            assert key in xi_dic.keys()

        # Encode input image
        im_hid_dic, enc_pts_dic = {}, {}
        for key in self.IMG_KEYS:
            xi = xi_dic[key]
            im_hid_dic[key] = self.im_encoders[key](xi)
            enc_pts, _ = self.pos_encoders[key](xi)
            enc_pts_dic[key] = enc_pts.reshape(-1, self.k_dim * 2)

        # Reshape encoded points and concatenate with input vector
        hid = torch.cat([enc_pts_dic["bp"], enc_pts_dic["inhand"], obs], -1)

        # LSTM forward pass
        new_rnn_state = self.rec(hid, rnn_state)

        # Decode joint prediction
        y_act = self.decoder_action(new_rnn_state[0])

        # Decode points
        dec_pts = self.decoder_point(new_rnn_state[0])

        # Reshape decoded points
        bp_dec_pts, inhand_dec_pts = torch.chunk(dec_pts, 2, dim=-1)
        dec_pts_dic = {"bp": bp_dec_pts, "inhand": inhand_dec_pts}
        yi_dic = {}
        for key in self.IMG_KEYS:
            dec_pts_in = dec_pts_dic[key].reshape(-1, self.k_dim, 2)

            # Inverse Spatial Softmax
            heatmap = self.issms[key](dec_pts_in)

            # Multiply heatmap with image feature `im_hid`
            hid_in = torch.mul(heatmap, im_hid_dic[key])

            # Decode image
            yi_dic[key] = self.decoder_images[key](hid_in)

        return yi_dic, y_act, enc_pts_dic, dec_pts_dic, new_rnn_state

    def get_params(self):
        return self.parameters()
