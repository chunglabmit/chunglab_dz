import setuptools

setuptools.setup(
    name="chunglab_dz",
    author="Kwanghun Chung Lab",
    description="Blockfs to deep zoom translator",
    packages=["chunglab_dz"],
    entry_points=dict(console_scripts=["chunglab-dz=chunglab_dz.main:main"]),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.5"
    ],
    install_requires=[
        "blockfs",
        "nuggt",
        "pandas",
        "DeepZoomTools",
        "tifffile"
    ]

)