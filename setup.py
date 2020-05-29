from setuptools import setup, find_packages

required_packages = ['numpy', 'torch', 'easydict', 'SimpleITK']


setup(name='detection3d',
    version='1.0',
    description='3D Medical Image Detection Toolkit, including landmark detection and object detection.',
    packages=find_packages(),
    url='https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit',
    author='IDEA Lab, the University of North Carolina at Chapel Hill.',
    author_email='qinliu19@email.unc.edu',
    license='GNU GENERAL PUBLIC LICENSE V3.0',
    install_requires=required_packages,
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts':
            ['obj_det_train=detection3d.obj_det_train:main',
             'obj_det_infer=detection3d.obj_det_infer:main',
             'lmk_det_train=detection3d.lmk_det_train:main',
             'lmk_det_infer=detection3d.lmk_det_infer:main']
    }
)