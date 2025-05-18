@daifu.with_goto
def main_daifu_rest_2():
    try:
        global i, inputs, outputs, x
        goto .restart
        label .restart
        cell_1()
        outputs = model(**inputs)
    except Exception as main_exception_2:
        daifu.CT_MANAGER.save(locals())
        raise
    else:
        return None, None
