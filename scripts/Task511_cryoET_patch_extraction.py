import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from skimage.morphology import label


def find_coms(img: np.ndarray):
    components, n_comp = label(img, 0, True)
    coms = []
    for comp in range(1, n_comp + 1):
        mask = components == comp
        coords = np.array(np.where(mask))
        center = np.round(np.mean(coords, 1)).astype(int)
        coms.append(center)
    return coms


if __name__ == "__main__":

    patchloc_base = "./patch_centers"

    # RAW TOMOGRAMS
    # tomos_base = '/home/haicu/jeronimo.carvajal/Thesis/data/raw_cryo-ET'
    # out_dir = '/home/haicu/jeronimo.carvajal/Thesis/data/raw_cryo-ET/patch_creation/result'

    # CRYOCARE+ISONET
    # tomos_base = '/home/haicu/jeronimo.carvajal/Thesis/data/isoNet/cryoCARE_corrected'
    # out_dir = '/home/haicu/jeronimo.carvajal/Thesis/data/isoNet/cryoCARE_corrected/patch_creation/result'

    # ISONET
    # tomos_base = (
    #     "/home/haicu/jeronimo.carvajal/Thesis/data/isoNet/RAW_dataset/RAW_corrected_i30"
    # )

    # F2F denoised
    tomos_base = (
        "/home/ubuntu/Thesis/data/S2SDenoising/F2FDenoised"
    )
    out_dir = tomos_base + "/patch_creation/result"

    patch_size = (128, 128, 128)
    padding = (16, 16, 16)
    padded_patch_size = (160, 160, 160)

    # now sample patches
    task_id = 511
    task_name = "cryoET"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(out_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    maybe_mkdir_p(imagestr)

    # loop over different patch iterations since we haven't extracted all at once
    for patch_iter in [1, 2, 3]:
        p_iter = str(patch_iter)

        patchloc_base_iter = join(patchloc_base, p_iter)

        if p_iter == "2":  # did not add patches to tomo17 in patcher iteration 2
            tomo_ids = [2, 3, 4, 10, 32, 38]
        else:
            tomo_ids = [2, 3, 4, 10, 17, 32, 38]
        for t in tomo_ids:
            print(t)
            # raw tomogram
            # name = 'tomo%02.0d.mrc'

            # CRYOCARE+ISONET
            # name = 'tomo%02.0d_bin4_denoised_0000_corrected.mrc'

            # ISONET
            # name = "tomo%02.0d_corrected.mrc"

            # F2F denoised
            name = "tomo%02.0d_s2sDenoised.mrc"

            image = sitk.GetArrayFromImage(sitk.ReadImage(join(tomos_base, name % t)))
            image = image - image.mean()
            image = image / image.std()

            # image = sitk.GetArrayFromImage(sitk.ReadImage(join(tomos_base, 'tomo%02.0d_bin4_denoised_0000.nii.gz' % t)))
            patchlocs = sitk.GetArrayFromImage(
                sitk.ReadImage(
                    join(patchloc_base_iter, "tomo%02.0d_patch_centers.nii.gz" % t)
                )
            )

            coms = find_coms(patchlocs)
            # now sample patches around COM
            for ctr, c in enumerate(coms):
                bbox = [
                    [c[i] - padded_patch_size[i] // 2, c[i] + padded_patch_size[i] // 2]
                    for i in range(3)
                ]
                for i in range(3):
                    mn = min(bbox[i])
                    if mn < 0:
                        bbox[i] = [j - mn for j in bbox[i]]
                for i in range(3):
                    mx = max(bbox[i])
                    if mx > image.shape[i]:
                        diff = mx - image.shape[i]
                        bbox[i] = [j - diff for j in bbox[i]]
                slices = tuple([slice(*i) for i in bbox])
                slices_center = tuple(
                    [
                        slice(i[0] + padding[j], i[1] - padding[j])
                        for j, i in enumerate(bbox)
                    ]
                )
                i = image[slices]

                if p_iter == "1":
                    numbering = ctr
                elif p_iter == "2":
                    numbering = ctr + 20
                elif p_iter == "3":
                    numbering = ctr + 30
                my_name = "tomo%02.0d_patch%03.0d" % (t, numbering)

                sitk.WriteImage(
                    sitk.GetImageFromArray(i), join(imagestr, my_name + "_0000.nii.gz")
                )
