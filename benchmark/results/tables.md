
### big_assembly  lin=0.25  (865,462 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 5.6108 s | 5.4433 | 5.7501 | 6.062 | 1.08 | 492 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 4.6670 s | 4.5935 | 4.6701 | 4.594 | 0.98 | 317 MB | 1.20x |
| pySMESH (parallel) **TRI MISMATCH** | 1.1397 s | 1.0490 | 1.1967 | 9.766 | 8.57 | 360 MB | 4.92x |
| pySMESH (stateless) **TRI MISMATCH** | 1.7865 s | 1.7805 | 1.8209 | 12.031 | 6.73 | 350 MB | 3.14x |

### big_assembly  lin=1  (302,358 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 2.8150 s | 2.7830 | 2.9116 | 2.984 | 1.06 | 344 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 2.1858 s | 2.1583 | 2.1960 | 2.125 | 0.97 | 300 MB | 1.29x |
| pySMESH (parallel) **TRI MISMATCH** | 0.9731 s | 0.8992 | 1.0000 | 5.562 | 5.72 | 300 MB | 2.89x |
| pySMESH (stateless) **TRI MISMATCH** | 1.6634 s | 1.5904 | 1.6958 | 7.812 | 4.70 | 279 MB | 1.69x |

### block_533f  lin=0.05  (14,174 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.1996 s | 0.1954 | 0.2496 | 0.250 | 1.25 | 77 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.1174 s | 0.1142 | 0.1194 | 0.125 | 1.07 | 75 MB | 1.70x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0612 s | 0.0583 | 0.0618 | 0.375 | 6.13 | 81 MB | 3.26x |
| pySMESH (stateless) **TRI MISMATCH** | 0.0978 s | 0.0954 | 0.0999 | 0.547 | 5.59 | 81 MB | 2.04x |

### fusee_249f  lin=0.1  (13,225 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.1403 s | 0.1329 | 0.1464 | 0.141 | 1.00 | 82 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.0966 s | 0.0956 | 0.0990 | 0.109 | 1.13 | 81 MB | 1.45x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0399 s | 0.0390 | 0.0413 | 0.312 | 7.84 | 81 MB | 3.52x |
| pySMESH (stateless) **TRI MISMATCH** | 0.0695 s | 0.0672 | 0.0715 | 0.359 | 5.17 | 81 MB | 2.02x |

### kurbelwelle_58f  lin=0.02  (2,490 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0302 s | 0.0289 | 0.0318 | 0.031 | 1.03 | 57 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0182 s | 0.0161 | 0.0193 | 0.031 | 1.72 | 59 MB | 1.66x |
| pySMESH (parallel) | 0.0084 s | 0.0080 | 0.0109 | 0.016 | 1.87 | 64 MB | 3.61x |
| pySMESH (stateless) | 0.0139 s | 0.0121 | 0.0149 | 0.016 | 1.12 | 64 MB | 2.17x |

### laufrad_123f  lin=0.2  (47,071 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.2515 s | 0.2495 | 0.2592 | 0.266 | 1.06 | 88 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.2186 s | 0.2118 | 0.2286 | 0.234 | 1.07 | 91 MB | 1.15x |
| pySMESH (parallel) **TRI MISMATCH** | 0.1775 s | 0.1703 | 0.1827 | 0.328 | 1.85 | 96 MB | 1.42x |
| pySMESH (stateless) **TRI MISMATCH** | 0.1810 s | 0.1773 | 0.1826 | 0.312 | 1.73 | 96 MB | 1.39x |

### part_26f  lin=0.05  (972 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0132 s | 0.0123 | 0.0143 | 0.031 | 2.36 | 55 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0113 s | 0.0106 | 0.0133 | 0.016 | 1.38 | 59 MB | 1.17x |
| pySMESH (parallel) | 0.0076 s | 0.0067 | 0.0088 | 0.031 | 4.10 | 64 MB | 1.74x |
| pySMESH (stateless) | 0.0087 s | 0.0081 | 0.0095 | 0.016 | 1.79 | 63 MB | 1.52x |

### spheres_1  lin=0.1  (978 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0030 s | 0.0028 | 0.0063 | 0.016 | 5.27 | 51 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0024 s | 0.0024 | 0.0025 | 0.016 | 6.54 | 56 MB | 1.24x |
| pySMESH (parallel) | 0.0025 s | 0.0025 | 0.0035 | 0.016 | 6.13 | 57 MB | 1.16x |
| pySMESH (stateless) | 0.0032 s | 0.0026 | 0.0052 | 0.016 | 4.94 | 56 MB | 0.94x |

### spheres_10  lin=0.1  (9,780 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0294 s | 0.0274 | 0.0319 | 0.031 | 1.06 | 54 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0219 s | 0.0206 | 0.0242 | 0.031 | 1.43 | 58 MB | 1.34x |
| pySMESH (parallel) | 0.0062 s | 0.0060 | 0.0065 | 0.016 | 2.53 | 66 MB | 4.76x |
| pySMESH (stateless) | 0.0067 s | 0.0056 | 0.0075 | 0.078 | 11.63 | 67 MB | 4.37x |

### spheres_100  lin=0.1  (97,800 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.2755 s | 0.2643 | 0.2782 | 0.266 | 0.96 | 85 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.2162 s | 0.2129 | 0.2243 | 0.219 | 1.01 | 69 MB | 1.27x |
| pySMESH (parallel) | 0.0334 s | 0.0319 | 0.0369 | 0.250 | 7.49 | 81 MB | 8.26x |
| pySMESH (stateless) | 0.0390 s | 0.0387 | 0.0408 | 0.281 | 7.21 | 81 MB | 7.06x |

### spheres_1000  lin=0.1  (978,000 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 2.7282 s | 2.6902 | 2.8623 | 2.656 | 0.97 | 381 MB | 1.00x (ref) |
| pySMESH (1 thread) | 2.1768 s | 2.1632 | 2.1937 | 2.141 | 0.98 | 185 MB | 1.25x |
| pySMESH (parallel) | 0.3103 s | 0.3036 | 0.3298 | 3.250 | 10.47 | 193 MB | 8.79x |
| pySMESH (stateless) | 0.3640 s | 0.3551 | 0.3785 | 3.562 | 9.79 | 183 MB | 7.49x |

### spheres_5000  lin=0.1  (4,890,000 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 13.7830 s | 13.6994 | 14.1668 | 13.797 | 1.00 | 1693 MB | 1.00x (ref) |
| pySMESH (1 thread) | 10.9276 s | 10.8636 | 11.0830 | 10.703 | 0.98 | 709 MB | 1.26x |
| pySMESH (parallel) | 1.5684 s | 1.5435 | 1.5987 | 16.531 | 10.54 | 715 MB | 8.79x |
| pySMESH (stateless) | 1.8458 s | 1.8076 | 1.8615 | 18.078 | 9.79 | 659 MB | 7.47x |

### sweep_block  lin=0.005  (44,117 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.3358 s | 0.3265 | 0.3576 | 0.344 | 1.02 | 90 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.2430 s | 0.2369 | 0.2527 | 0.250 | 1.03 | 81 MB | 1.38x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0748 s | 0.0739 | 0.0756 | 0.734 | 9.82 | 92 MB | 4.49x |
| pySMESH (stateless) **TRI MISMATCH** | 0.1234 s | 0.1208 | 0.1249 | 0.812 | 6.58 | 91 MB | 2.72x |

### sweep_block  lin=0.02  (21,068 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.2233 s | 0.2113 | 0.2359 | 0.250 | 1.12 | 80 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.1446 s | 0.1416 | 0.1532 | 0.156 | 1.08 | 76 MB | 1.54x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0641 s | 0.0629 | 0.0685 | 0.516 | 8.05 | 84 MB | 3.49x |
| pySMESH (stateless) **TRI MISMATCH** | 0.1031 s | 0.1023 | 0.1038 | 0.594 | 5.76 | 84 MB | 2.17x |

### sweep_block  lin=0.05  (14,174 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.1913 s | 0.1883 | 0.1994 | 0.203 | 1.06 | 78 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.1195 s | 0.1125 | 0.1305 | 0.125 | 1.05 | 75 MB | 1.60x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0587 s | 0.0583 | 0.0593 | 0.344 | 5.86 | 81 MB | 3.26x |
| pySMESH (stateless) **TRI MISMATCH** | 0.0975 s | 0.0956 | 0.1054 | 0.516 | 5.29 | 81 MB | 1.96x |

### sweep_block  lin=0.2  (10,708 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.1795 s | 0.1769 | 0.1825 | 0.203 | 1.13 | 77 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.1060 s | 0.1041 | 0.1101 | 0.109 | 1.03 | 75 MB | 1.69x |
| pySMESH (parallel) | 0.0585 s | 0.0573 | 0.0598 | 0.438 | 7.48 | 79 MB | 3.07x |
| pySMESH (stateless) | 0.0947 s | 0.0937 | 0.0950 | 0.422 | 4.45 | 80 MB | 1.89x |

### sweep_block  lin=0.5  (10,480 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.1735 s | 0.1680 | 0.1819 | 0.188 | 1.08 | 75 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.1016 s | 0.1001 | 0.1074 | 0.109 | 1.08 | 75 MB | 1.71x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0590 s | 0.0563 | 0.0604 | 0.438 | 7.42 | 80 MB | 2.94x |
| pySMESH (stateless) **TRI MISMATCH** | 0.0977 s | 0.0957 | 0.0992 | 0.500 | 5.12 | 80 MB | 1.78x |

### sweep_cone  lin=0.003  (6,956 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0943 s | 0.0881 | 0.1043 | 0.094 | 0.99 | 80 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0968 s | 0.0899 | 0.1054 | 0.094 | 0.97 | 85 MB | 0.97x |
| pySMESH (parallel) | 0.0906 s | 0.0873 | 0.0956 | 0.109 | 1.21 | 86 MB | 1.04x |
| pySMESH (stateless) | 0.0969 s | 0.0926 | 0.1040 | 0.109 | 1.13 | 85 MB | 0.97x |

### sweep_cone  lin=0.01  (2,620 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0229 s | 0.0206 | 0.0248 | 0.031 | 1.36 | 58 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0204 s | 0.0183 | 0.0213 | 0.031 | 1.53 | 63 MB | 1.12x |
| pySMESH (parallel) | 0.0187 s | 0.0164 | 0.0224 | 0.031 | 1.67 | 64 MB | 1.23x |
| pySMESH (stateless) | 0.0195 s | 0.0175 | 0.0207 | 0.031 | 1.60 | 64 MB | 1.17x |

### sweep_cone  lin=0.03  (1,054 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0066 s | 0.0061 | 0.0076 | 0.016 | 2.38 | 53 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0060 s | 0.0058 | 0.0083 | 0.016 | 2.58 | 58 MB | 1.09x |
| pySMESH (parallel) | 0.0054 s | 0.0053 | 0.0095 | 0.016 | 2.88 | 59 MB | 1.21x |
| pySMESH (stateless) | 0.0062 s | 0.0060 | 0.0078 | 0.016 | 2.54 | 58 MB | 1.07x |

### sweep_cone  lin=0.1  (458 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0028 s | 0.0026 | 0.0031 | 0.016 | 5.58 | 52 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0023 s | 0.0023 | 0.0040 | 0.016 | 6.84 | 56 MB | 1.22x |
| pySMESH (parallel) | 0.0032 s | 0.0023 | 0.0037 | 0.016 | 4.91 | 57 MB | 0.88x |
| pySMESH (stateless) | 0.0027 s | 0.0024 | 0.0043 | 0.016 | 5.88 | 57 MB | 1.05x |

### sweep_cone  lin=0.3  (214 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0020 s | 0.0017 | 0.0026 | 0.016 | 7.66 | 52 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0014 s | 0.0014 | 0.0022 | 0.016 | 10.91 | 57 MB | 1.43x |
| pySMESH (parallel) | 0.0018 s | 0.0013 | 0.0021 | 0.016 | 8.87 | 57 MB | 1.16x |
| pySMESH (stateless) | 0.0022 s | 0.0017 | 0.0034 | 0.000 | 0.00 | 57 MB | 0.94x |

### sweep_cone  lin=1  (152 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0015 s | 0.0015 | 0.0027 | 0.016 | 10.36 | 51 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0013 s | 0.0012 | 0.0032 | 0.016 | 12.44 | 57 MB | 1.20x |
| pySMESH (parallel) | 0.0017 s | 0.0012 | 0.0019 | 0.000 | 0.00 | 57 MB | 0.90x |
| pySMESH (stateless) | 0.0017 s | 0.0015 | 0.0022 | 0.000 | 0.00 | 57 MB | 0.88x |

### sweep_sphere1  lin=0.003  (33,506 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.2573 s | 0.2515 | 0.2651 | 0.266 | 1.03 | 111 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.2449 s | 0.2306 | 0.2488 | 0.250 | 1.02 | 116 MB | 1.05x |
| pySMESH (parallel) | 0.2700 s | 0.2502 | 0.2727 | 0.281 | 1.04 | 117 MB | 0.95x |
| pySMESH (stateless) | 0.2502 s | 0.2372 | 0.2601 | 0.250 | 1.00 | 116 MB | 1.03x |

### sweep_sphere1  lin=0.01  (10,108 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0427 s | 0.0397 | 0.0434 | 0.062 | 1.47 | 63 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0365 s | 0.0347 | 0.0412 | 0.047 | 1.28 | 68 MB | 1.17x |
| pySMESH (parallel) | 0.0408 s | 0.0395 | 0.0415 | 0.047 | 1.15 | 68 MB | 1.04x |
| pySMESH (stateless) | 0.0354 s | 0.0339 | 0.0399 | 0.047 | 1.32 | 68 MB | 1.20x |

### sweep_sphere1  lin=0.03  (3,278 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0099 s | 0.0098 | 0.0104 | 0.016 | 1.58 | 54 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0085 s | 0.0084 | 0.0126 | 0.016 | 1.85 | 59 MB | 1.17x |
| pySMESH (parallel) | 0.0088 s | 0.0084 | 0.0118 | 0.016 | 1.78 | 59 MB | 1.13x |
| pySMESH (stateless) | 0.0100 s | 0.0087 | 0.0120 | 0.016 | 1.56 | 59 MB | 0.99x |

### sweep_sphere1  lin=0.1  (978 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0029 s | 0.0028 | 0.0032 | 0.016 | 5.40 | 51 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0025 s | 0.0024 | 0.0025 | 0.000 | 0.00 | 56 MB | 1.18x |
| pySMESH (parallel) | 0.0025 s | 0.0024 | 0.0028 | 0.016 | 6.37 | 57 MB | 1.18x |
| pySMESH (stateless) | 0.0028 s | 0.0027 | 0.0033 | 0.016 | 5.60 | 56 MB | 1.04x |

### sweep_sphere1  lin=0.3  (360 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0015 s | 0.0013 | 0.0018 | 0.016 | 10.54 | 51 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0011 s | 0.0011 | 0.0014 | 0.016 | 13.69 | 56 MB | 1.30x |
| pySMESH (parallel) | 0.0014 s | 0.0013 | 0.0023 | 0.016 | 10.84 | 56 MB | 1.03x |
| pySMESH (stateless) | 0.0014 s | 0.0013 | 0.0015 | 0.016 | 11.31 | 56 MB | 1.07x |

### sweep_sphere1  lin=1  (306 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0013 s | 0.0013 | 0.0017 | 0.016 | 12.00 | 51 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.0010 s | 0.0010 | 0.0016 | 0.016 | 15.07 | 56 MB | 1.26x |
| pySMESH (parallel) | 0.0012 s | 0.0011 | 0.0017 | 0.016 | 13.32 | 56 MB | 1.11x |
| pySMESH (stateless) | 0.0014 s | 0.0013 | 0.0020 | 0.016 | 11.47 | 56 MB | 0.96x |

### sweep_spheres1000  lin=0.01  (10,108,000 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 40.8554 s | 39.8415 | 41.0686 | 40.219 | 0.98 | 3266 MB | 1.00x (ref) |
| pySMESH (1 thread) | 34.7289 s | 34.7061 | 34.8242 | 34.156 | 0.98 | 1238 MB | 1.18x |
| pySMESH (parallel) | 4.9929 s | 4.8927 | 5.2346 | 60.750 | 12.17 | 1282 MB | 8.18x |
| pySMESH (stateless) | 5.0201 s | 4.8424 | 5.3459 | 59.609 | 11.87 | 1201 MB | 8.14x |

### sweep_spheres1000  lin=0.03  (3,278,000 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 10.2797 s | 10.2056 | 10.4105 | 10.188 | 0.99 | 1096 MB | 1.00x (ref) |
| pySMESH (1 thread) | 8.7186 s | 8.6400 | 8.7505 | 8.453 | 0.97 | 451 MB | 1.18x |
| pySMESH (parallel) | 1.0978 s | 1.0891 | 1.1095 | 12.219 | 11.13 | 465 MB | 9.36x |
| pySMESH (stateless) | 1.1453 s | 1.1362 | 1.2281 | 12.750 | 11.13 | 437 MB | 8.98x |

### sweep_spheres1000  lin=0.1  (978,000 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 2.7016 s | 2.6987 | 2.7715 | 2.703 | 1.00 | 381 MB | 1.00x (ref) |
| pySMESH (1 thread) | 2.1854 s | 2.1399 | 2.2013 | 2.141 | 0.98 | 185 MB | 1.24x |
| pySMESH (parallel) | 0.3077 s | 0.3036 | 0.3117 | 3.203 | 10.41 | 192 MB | 8.78x |
| pySMESH (stateless) | 0.3626 s | 0.3544 | 0.4198 | 3.453 | 9.52 | 183 MB | 7.45x |

### sweep_spheres1000  lin=0.3  (360,000 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 1.1531 s | 1.1241 | 1.1891 | 1.172 | 1.02 | 204 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.8725 s | 0.8474 | 0.8847 | 0.859 | 0.98 | 114 MB | 1.32x |
| pySMESH (parallel) | 0.1600 s | 0.1559 | 0.1672 | 1.609 | 10.06 | 119 MB | 7.21x |
| pySMESH (stateless) | 0.2099 s | 0.2073 | 0.2225 | 1.781 | 8.49 | 116 MB | 5.49x |

### sweep_spheres1000  lin=1  (306,000 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 1.0052 s | 0.9920 | 1.0397 | 1.062 | 1.06 | 187 MB | 1.00x (ref) |
| pySMESH (1 thread) | 0.7527 s | 0.7401 | 0.7720 | 0.750 | 1.00 | 107 MB | 1.34x |
| pySMESH (parallel) | 0.1496 s | 0.1455 | 0.1505 | 1.359 | 9.09 | 115 MB | 6.72x |
| pySMESH (stateless) | 0.1975 s | 0.1948 | 0.2004 | 1.672 | 8.46 | 111 MB | 5.09x |

### ujoint_133f  lin=0.05  (9,602 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0838 s | 0.0802 | 0.0863 | 0.094 | 1.12 | 67 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.0545 s | 0.0527 | 0.0557 | 0.062 | 1.15 | 67 MB | 1.54x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0206 s | 0.0204 | 0.0219 | 0.203 | 9.85 | 71 MB | 4.06x |
| pySMESH (stateless) **TRI MISMATCH** | 0.0385 s | 0.0377 | 0.0403 | 0.188 | 4.87 | 71 MB | 2.18x |

### wrenchnut_36f  lin=0.05  (24,936 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.1477 s | 0.1390 | 0.1510 | 0.156 | 1.06 | 64 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.1300 s | 0.1275 | 0.1325 | 0.141 | 1.08 | 65 MB | 1.14x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0331 s | 0.0324 | 0.0347 | 0.156 | 4.72 | 81 MB | 4.46x |
| pySMESH (stateless) **TRI MISMATCH** | 0.0396 s | 0.0375 | 0.0421 | 0.203 | 5.12 | 83 MB | 3.73x |

### zylkopf_137f  lin=0.02  (6,230 triangles)
| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmsh | 0.0659 s | 0.0629 | 0.0677 | 0.078 | 1.19 | 61 MB | 1.00x (ref) |
| pySMESH (1 thread) **TRI MISMATCH** | 0.0380 s | 0.0357 | 0.0399 | 0.047 | 1.23 | 63 MB | 1.74x |
| pySMESH (parallel) **TRI MISMATCH** | 0.0179 s | 0.0174 | 0.0201 | 0.125 | 6.98 | 69 MB | 3.68x |
| pySMESH (stateless) **TRI MISMATCH** | 0.0288 s | 0.0281 | 0.0299 | 0.031 | 1.09 | 70 MB | 2.29x |

wrote D:\projects\pysmesh\benchmark\results\summary.csv
