#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central Configuration for 3D Data Processing
=============================================

Description:
    This file centralizes all dataset-specific configurations for stable,
    physical parameters like the camera's mask center.

Usage:
    

    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])
    mask_center = config['mask_center']
"""
import numpy as np

# A dictionary to hold parameters for each dataset.
# 'mask_center' defines the center of the circular mask for the endoscope view.
DATASET_CONFIGS = {
    "default": {
        "mask_center": (1040, 550),
    },
    "dataset1": {
        "mask_center": (1025, 560),
    },
    "dataset2": {
        "mask_center": (1020, 555),
    },
    "dataset3": {
        "mask_center": (1028, 540),
    },
    "dataset4": {
        "mask_center": (1028, 545),
    },
    "dataset5": {
        "mask_center": (1025, 555),
    },
    "dataset6": {
        "mask_center": (1025, 548),
    },
    "dataset7": {
        "mask_center": (1025, 545),
    },
    "dataset8": {
        "mask_center": (1025, 555),
    },
    "dataset9": {
        "mask_center": (1025, 550),
    },
    "dataset10": {
        "mask_center": (1025, 555),
    },
    "dataset11": {
        "mask_center": (1025, 555),
    },
    "dataset12": {
        "mask_center": (1025, 555),
    },
    "dataset13": {
        "mask_center": (1025, 555),
    },
    "dataset14": {
        "mask_center": (1025, 555),
    },
    "dataset15": {
        "mask_center": (1025, 555),
    },
    "dataset16": {
        "mask_center": (1025, 545),
    },
    "dataset17": {
        "mask_center": (1028, 555),
    },
    "dataset18": {
        "mask_center": (1028, 555),
    },
    "dataset19": {
        "mask_center": (1028, 550),
    },
    "dataset20": {
        "mask_center": (1028, 550),
    },
    "dataset21": {
        "mask_center": (1025, 550),
    },
    "dataset22": {
        "mask_center": (1028, 550),
    },
    "dataset23": {
        "mask_center": (1028, 550),
    },
    "dataset24": {
        "mask_center": (1028, 555),
    },
    "dataset25": {
        "mask_center": (1028, 550),
    },
    "dataset26": {
        "mask_center": (1028, 550),
    },
    "dataset27": {
        "mask_center": (1028, 550),
    },
}