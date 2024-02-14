import copy
import numpy as np
import gradio as gr
from PIL import Image
from modules import scripts_postprocessing
from modules import shared, errors
if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None


NAME = 'ControlNet Preprocessor'


g_cn_modules = None
def getCNModules():
    global g_cn_modules
    if g_cn_modules is None:
        import scripts.global_state
        g_cn_modules = copy.copy(scripts.global_state.cn_preprocessor_modules)
        inpaintKeys = []
        for key in g_cn_modules:
            if key.startswith('inpaint'):
                inpaintKeys.append(key)
        for key in inpaintKeys:
            del g_cn_modules[key]
    return g_cn_modules


g_pixel_perfect_resolution = None
g_resize_mode = None
def getPixelPerfectResolution(
        image: np.ndarray,
        target_H: int,
        target_W: int):
    global g_pixel_perfect_resolution, g_resize_mode
    if g_pixel_perfect_resolution is None:
        from scripts import external_code
        g_pixel_perfect_resolution = external_code.pixel_perfect_resolution
        g_resize_mode = external_code.ResizeMode.RESIZE

    return g_pixel_perfect_resolution(
        image, target_H=target_H, target_W=target_W, resize_mode=g_resize_mode)


g_cn_HWC3 = None
def convertIntoCNImageFromat(image):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        from annotator.util import HWC3
        g_cn_HWC3 = HWC3

    color = g_cn_HWC3(np.asarray(image).astype(np.uint8))
    return color


def convertImageIntoPILFormat(image):
    return Image.fromarray(
        np.ascontiguousarray(image.clip(0, 255).astype(np.uint8)).copy()
    )


def get_default_ui_unit(is_ui=True):
    from scripts.controlnet_ui.controlnet_ui_group import UiControlNetUnit
    from scripts import external_code
    cls = UiControlNetUnit if is_ui else external_code.ControlNetUnit
    return cls(
        enabled=False,
        module="none",
        model="None"
    )

class CNInExtrasTab(scripts_postprocessing.ScriptPostprocessing):
    name = NAME
    order = 18000

    def ui(self):
        try:
            self.default_unit = get_default_ui_unit()
            with (
                InputAccordion(False, label=NAME) if InputAccordion
                else gr.Accordion(NAME, open=False)
                as self.enable
            ):
                if not InputAccordion:
                    self.enable = gr.Checkbox(False, label="Enable")
                with gr.Row():
                    modulesList = list(getCNModules().keys())
                    self.module = gr.Dropdown(modulesList, label="Module", value=modulesList[0])
                    self.pixel_perfect = gr.Checkbox(
                        label="Pixel Perfect",
                        value=True,
                        elem_id=f"extras_controlnet_pixel_perfect_checkbox",
                    )
                with gr.Row():
                    self.create_sliders()
            self.register_build_sliders()
            args = {
                'enable': self.enable,
                'module': self.module,
                'pixel_perfect': self.pixel_perfect,
                'processor_res' : self.processor_res,
                'threshold_a' : self.threshold_a,
                'threshold_b' : self.threshold_b,
            }
            return args
        except Exception as e:
            errors.report(f"Cannot init {NAME}", exc_info=True)
            return {}


    def create_sliders(self):
        # advanced options
        with gr.Column(visible=False) as self.advanced:
            self.processor_res = gr.Slider(
                label="Preprocessor resolution",
                value=self.default_unit.processor_res,
                minimum=64,
                maximum=2048,
                visible=False,
                interactive=True,
                elem_id=f"extras_controlnet_preprocessor_resolution_slider",
            )
            self.threshold_a = gr.Slider(
                label="Threshold A",
                value=self.default_unit.threshold_a,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"extras_controlnet_threshold_A_slider",
            )
            self.threshold_b = gr.Slider(
                label="Threshold B",
                value=self.default_unit.threshold_b,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"extras_controlnet_threshold_B_slider",
            )


    def register_build_sliders(self):
        from scripts.processor import (
            preprocessor_sliders_config,
            flag_preprocessor_resolution,
        )
        from scripts import global_state

        def build_sliders(module: str, pp: bool):
            
            # Clear old slider values so that they do not cause confusion in
            # infotext.
            clear_slider_update = gr.update(
                visible=False,
                interactive=True,
                minimum=-1,
                maximum=-1,
                value=-1,
            )

            grs = []
            module = global_state.get_module_basename(module)
            if module not in preprocessor_sliders_config:
                default_res_slider_config = dict(
                    label=flag_preprocessor_resolution,
                    minimum=64,
                    maximum=2048,
                    step=1,
                )

                default_res_slider_config["value"] = 512

                grs += [
                    gr.update(
                        **default_res_slider_config,
                        visible=not pp,
                        interactive=True,
                    ),
                    copy.copy(clear_slider_update),
                    copy.copy(clear_slider_update),
                    gr.update(visible=True),
                ]
            else:
                for slider_config in preprocessor_sliders_config[module]:
                    if isinstance(slider_config, dict):
                        visible = True
                        if slider_config["name"] == flag_preprocessor_resolution:
                            visible = not pp
                        slider_update = gr.update(
                            label=slider_config["name"],
                            minimum=slider_config["min"],
                            maximum=slider_config["max"],
                            step=slider_config["step"]
                            if "step" in slider_config
                            else 1,
                            visible=visible,
                            interactive=True,
                        )
                        slider_update["value"] = slider_config["value"]

                        grs.append(slider_update)

                    else:
                        grs.append(copy.copy(clear_slider_update))
                while len(grs) < 3:
                    grs.append(copy.copy(clear_slider_update))
                grs.append(gr.update(visible=True))

            return grs

        inputs = [
            self.module,
            self.pixel_perfect,
        ]
        outputs = [
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.advanced,
        ]
        self.module.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )

        self.pixel_perfect.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )



    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if args['enable'] == False:
            return
        
        w, h = pp.image.size
        image = convertIntoCNImageFromat(pp.image)

        if args['pixel_perfect']:
            processor_res = getPixelPerfectResolution(
                image,
                target_H=h,
                target_W=w,
            )
        else:
            processor_res = args['processor_res']

        detected_map, is_image = getCNModules()[args['module']](
            image,
            res=processor_res,
            thr_a=args['threshold_a'],
            thr_b=args['threshold_b'],
            low_vram=shared.cmd_opts.lowvram,
        )

        if is_image:
            pp.image = convertImageIntoPILFormat(detected_map)
            info = copy.copy(args)
            del info['enable']
            if info['pixel_perfect']:
                info['processor_res'] = processor_res
            del info['pixel_perfect']
            pp.info[NAME] = str(info)
