from skimage import measure
import numpy as np

# filename should be the full path to a .npy file with 2D numpy array in it
# function will return a binary counting image derived from the input image
def counting_frame(filename):

    IntThreshold = 20    # intensity threshold for a single pixel to be considered electron hit
    SEnumber = 0
    SEint = 0

    frame = np.load(filename)
    frame_binary = np.heaviside(frame - IntThreshold,1)
    frame_counting = np.zeros(frame_binary.shape)
    all_labels = measure.label(frame_binary)

    # Determine number of total events
    ncomponents = np.amax(all_labels)

    for i in range(ncomponents):
        component_size = np.where(all_labels == i)[0].shape
        component_idx = np.where(all_labels == i)

        # append total intensity of this SE event to the total
        int_list = frame[component_idx[0],component_idx[1]]
        SEint = SEint + np.sum(int_list)
        SEnumber = SEnumber + 1

        # calculate COM for this event and assign that pixel to 1
        cor_row = np.average(component_idx[0],weights=int_list)
        cor_col = np.average(component_idx[1],weights=int_list)
        frame_counting[int(cor_row), int(cor_col)] = 1

    return(frame_counting)