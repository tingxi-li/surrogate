def main_daifu_cell_2():
    try:
        global i, inputs, outputs, x
        cell_1()
        outputs = model(**inputs)
    except Exception as main_exception_2:
        daifu.CT_MANAGER.save(locals())
        raise
    else:
        return None, None
