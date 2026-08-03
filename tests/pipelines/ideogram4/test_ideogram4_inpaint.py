# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from diffusers import Ideogram4InpaintPipeline


class Ideogram4InpaintPipelineTests(unittest.TestCase):
    def test_get_timesteps(self):
        timesteps = torch.arange(1000, 0, -100)

        class Scheduler:
            order = 1
            timesteps = timesteps

            def set_begin_index(self, begin_index):
                self.begin_index = begin_index

        pipe = Ideogram4InpaintPipeline.__new__(Ideogram4InpaintPipeline)
        pipe.scheduler = Scheduler()

        selected_timesteps, adjusted_steps, t_start = Ideogram4InpaintPipeline.get_timesteps(pipe, 10, 0.6)

        self.assertEqual(adjusted_steps, 6)
        self.assertEqual(t_start, 4)
        self.assertEqual(pipe.scheduler.begin_index, 4)
        self.assertTrue(torch.equal(selected_timesteps, timesteps[4:]))

    def test_prepare_mask_latents(self):
        pipe = Ideogram4InpaintPipeline.__new__(Ideogram4InpaintPipeline)
        mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

        mask = Ideogram4InpaintPipeline.prepare_mask_latents(
            pipe,
            mask=mask,
            batch_size=1,
            num_images_per_prompt=2,
            grid_h=2,
            grid_w=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertEqual(mask.shape, (2, 4, 1))
        self.assertTrue(torch.equal(mask[0], mask[1]))
        self.assertTrue(torch.equal(mask[0, :, 0], torch.tensor([0.0, 1.0, 1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
