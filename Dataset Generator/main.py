import numpy as np
from LPDataset import LicensePlateDataset
from LP import PlateType

def main():
    # np.random.seed(42)
    LPDS = LicensePlateDataset()

    # Generate synthetic data using background images from the resources folder
    # LPDS.generate(20, verbose=True)

    # Generate synthetic data using an existing dataset with YOLO annotations
    # Folder structure should be: /images and /labels
    # LPDS.generate_from_preannotated(r'D:\AutoVisionPR Data\Datasets\MyDatasetV3\Standard',
    #                                 per_image=4, verbose=True)
    # LPDS.generate_from_preannotated(r'D:\AutoVisionPR Data\Datasets\MyDatasetV3\Big',
    #                                 per_image=4, plate_type=PlateType.MOTORCYCLE, verbose=True)

    # Generate synthetic data using images from the CCPD dataset
    # LPDS.generate_from_ccpd('F:/Downloads/Datasets/CCPD2019/ccpd_base', n=150, per_image=2)


if __name__ == '__main__':
    main()
