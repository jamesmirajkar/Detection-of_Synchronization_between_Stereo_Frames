# Detection of Synchronization between Stereo Frames
This repository contains an algorithm for detecting the synchronization state of stereo frames and analyzing spatial information. The algorithm consists of two primary parts: the timestamp algorithm and the spatial algorithm. By accurately determining the synchronization state of stereo frames, this algorithm enhances various computer vision applications such as object detection, picture matching, and 3D reconstruction. As shown in the image a) represents stereo synchronous image whereas b) represents stereo asynchronous image
![Intro_img (1)](https://github.com/fardinkhanz/Detection-of-Synchronization-between-Stereo-Frames/assets/89691395/a3a48ec1-0c76-4f05-94ee-5a45bf20062a)


## Usage
### The algorithm can be used as follows:

Timestamp Algorithm: The timestamp algorithm examines the timestamps and frame IDs of the left and right frames. It takes into consideration a threshold determined by the video's frames-per-second (FPS) rate. The frames are considered synchronized if the absolute discrepancies between these values are less than the threshold; otherwise, they are regarded as asynchronous.


Spatial Algorithm: If the Timestamp Algorithm identifies a possible synchronization, the spatial algorithm is run. This algorithm generates a template using the Sobel filter and extracts spatial information from the left frame. It then matches the template to the right, right+1, and right-1 frames using an appropriate correlation approach. If the chosen frame has the highest correlation, the frames are considered synchronized. However, if the frame with the highest correlation is either the right+1 or right-1 frame, the frames are considered asynchronous.
![final_design (1)](https://github.com/fardinkhanz/Detection-of-Synchronization-between-Stereo-Frames/assets/89691395/d005519a-944b-44dc-b3c0-0a6aa8a0640f)
## Evaluation
The suggested approach has been evaluated on both static and dynamic datasets, providing high accuracy for stereo frame synchronization. The accuracy achieved is 90.33% for the static dataset and 96.67% for the dynamic dataset.

## Contributing
Contributions to this project are welcome. If you have any ideas, improvements, or bug fixes, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. Feel free to use and modify the code according to the terms of the license.

## Acknowledgments
We would like to thank the contributors and researchers who have made significant contributions to the field of computer vision, as well as the open-source community for their valuable resources and libraries.
