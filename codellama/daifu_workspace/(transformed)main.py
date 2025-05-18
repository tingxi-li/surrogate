@daifu.transform()
def main(x):
    daifu_store = {}
    for daifu_dataname in daifu.TRANSFORM_REGISTRY['main']['dataname_list']:
        if daifu_dataname in daifu.TRANSFORM_REGISTRY['main']['local_variables_list'] and daifu_dataname in globals():
            daifu_store[daifu_dataname] = globals()[daifu_dataname]
        if daifu_dataname in locals():
            globals()[daifu_dataname] = locals()[daifu_dataname]
        elif daifu_dataname not in globals():
            globals()[daifu_dataname] = None
    try:
        daifu_return_tag, daifu_return_item = main_daifu_cell_1()
    except Exception:
        while True:
            try:
                daifu.RP_MANAGER.repair()
                daifu_return_tag, daifu_return_item = main_daifu_rest_1()
                break
            except Exception:
                continue
    for daifu_dataname in daifu_store:
        globals()[daifu_dataname] = daifu_store[daifu_dataname]
