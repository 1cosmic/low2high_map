
def filter_data(src_arr, corr_type, nodata_value, mask=None):
    log_message(f"\nПрименение коррекции типа {corr_type}")

    if corr_type == 'med3':
        # Медианная фильтрация
        # result = cv2.medianBlur(src_arr, 3)
        result = median_filter(src_arr, size=3)
        # result = median(src_arr, square(3))
    elif corr_type == 'med5':
        # Медианная фильтрация
        # result = cv2.medianBlur(src_arr, 5)
        result = median_filter(src_arr, size=5)
        # result = median(src_arr, square(5))
    elif corr_type == 's4':
        # Удаление одиночных пикселей через connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src_arr, connectivity=8)
        result = np.zeros_like(src_arr)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 4:  # Оставить только объекты больше 4 пикселей
                result[labels == i] = 1
    else:
        raise ValueError(f"unsupported corr_type={corr_type}")

    if mask is not None:
        result = np.where(mask, result, nodata_value)

    return result.astype(src_arr.dtype)



def save_clf_results(clf_results, ref_ds, output_path, accuracy, report, colors, nodata_value):
    log_message("\nСохранение результатов классификации")
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, 
                         ref_ds.RasterXSize, 
                         ref_ds.RasterYSize, 
                         1, gdal.GDT_Byte,
                         options=c_options)
    
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(clf_results)
    out_band.SetNoDataValue(nodata_value)
    
    if colors is not None:
        out_band.SetColorTable(colors)
    
    log_message("\tПостроение пирамид")
    out_ds.BuildOverviews("NEAR", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    out_ds.FlushCache()
    
    if accuracy is not None and report is not None:
        report_path = os.path.splitext(output_path)[0] + "_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("Classification report:\n")
            f.write(report)
        log_message(f"\tОтчет о точности сохранен в: {report_path}")
    
    log_message(f"\tРезультаты сохранены в: {output_path}")



def set_color_table(mask_pal, count_cl):
    # задать цветовую палитру в файле
    if mask_pal is None:
        return None

    colors = gdal.ColorTable()
    colors.SetColorEntry(0, (0, 0, 0, 0)) # Прозрачный
    
    palettes = {
        'red_mask': [(1, (254, 1, 1, 255))], # Красный
        'green_mask': [(1, (50, 200, 50, 255))], # Зеленый
        'adaptive': [
            (1, (254, 1, 1, 255)),        # Красный
            (2, (1, 254, 1, 255)),        # Зеленый
            (3, (254, 165, 1, 255)),      # Оранжевый
            (4, (254, 254, 1, 255)),      # Желтый
            (5, (1, 254, 254, 255)),      # Бирюзовый
            (6, (1, 1, 254, 255)),        # Синий
            (7, (128, 1, 128, 255)),      # Фиолетовый
            (8, (255, 105, 180, 255)),    # Розовый
            (9, (255, 215, 0, 255)),      # Золотой
            (10, (139, 69, 19, 255)),     # Коричневый
            (11, (0, 128, 128, 255)),     # Темно-бирюзовый
            (12, (75, 0, 130, 255)),      # Индиго
            (13, (255, 140, 0, 255)),     # Темно-оранжевый
            (14, (50, 205, 50, 255)),     # Лаймовый
            (15, (70, 130, 180, 255)),    # Стальной синий
            (16, (255, 99, 71, 255)),     # Томатный
            (17, (64, 224, 208, 255)),    # Бирюзовый светлый
            (18, (138, 43, 226, 255)),    # Сине-фиолетовый
            (19, (210, 105, 30, 255)),    # Шоколадный
            (20, (0, 191, 255, 255))      # Глубокий небесно-синий
        ]
    }
    
    if mask_pal in palettes:
        entries = palettes[mask_pal]
        if mask_pal == 'adaptive':
            entries = entries[:min(count_cl, len(entries))]
        
        for idx, color in entries:
            colors.SetColorEntry(idx, color)

    return colors

# Палитра:

# Apply:
# if palette:
#     ds_cor_reg = gdal.Open(cor_reg, gdal.GA_Update)
#     band_cor_reg = ds_cor_reg.GetRasterBand(1)
#     add_palette(band_cor_reg)
#     del band_cor_reg, ds_cor_reg


nodata_map6cl = 0

classes_map6cl = {
    # 0: {'name': 'NODATA', 'color': 'magenta'},
    1: {'name': 'Лесные земли', 'color': (1, 143, 51, 255)},
    2: {'name': 'Возделываемые земли', 'color': (252, 155, 1, 255)},
    3: {'name': 'Луга и пастбища', 'color': (254, 249, 1, 255)},
    4: {'name': 'Водно-болотные угодия', 'color': (31, 5, 254, 255)},
    5: {'name': 'Поселения', 'color': (220, 16, 16, 255)},
    6: {'name': 'Прочиe земли', 'color': (1, 247, 254, 255)}
}

def add_palette(band):

    # Добавить к каналу палитру
    color_table = gdal.ColorTable()
    for class_id in sorted(classes_map6cl.keys()):
        color = classes_map6cl[class_id]['color']
        name = classes_map6cl[class_id]['name']
        color_table.SetColorEntry(class_id, color)
        band.SetMetadataItem(f'CLASS_{class_id}_NAME', name)
    band.SetRasterColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.FlushCache()

