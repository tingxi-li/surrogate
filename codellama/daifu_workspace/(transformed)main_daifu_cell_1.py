def main_daifu_cell_1():
    try:
        global i, inputs, outputs, x
        print('=== now start ===')
        print('x value: ', x)
        for i in range(x):
            try:
                daifu_return_tag, daifu_return_item = main_daifu_cell_2()
            except Exception:
                while True:
                    try:
                        daifu.RP_MANAGER.repair()
                        daifu_return_tag, daifu_return_item = main_daifu_rest_2()
                        if daifu_return_tag is not None:
                            if daifu_return_tag == 'break':
                                break
                            else:
                                return daifu_return_tag, daifu_return_item
                        break
                    except Exception:
                        continue
                if daifu_return_tag == 'break':
                    break
        pass
    except Exception as main_exception_1:
        daifu.CT_MANAGER.save(locals())
        raise
    else:
        return None, None
