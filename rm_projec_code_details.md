### Binarization Code
```python
def binarization(org, grad_min, show=False, write_path=None, wait_key=0):
    grey = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    grad = gray_to_gradient(grey)        # get RoI with high gradient
    rec, binary = cv2.threshold(grad, grad_min, 255, cv2.THRESH_BINARY)    # enhance the RoI
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, (3, 3))  # remove small noisy holes
    if write_path is not None:
        cv2.imwrite(write_path, morph)
    if show:
        cv2.imshow('binary', morph)
        if wait_key is not None:
            cv2.waitKey(wait_key)
    return morph
```

###


### Block Detection COde
``` python
def component_detection(binary, min_obj_area,
                        line_thickness=C.THRESHOLD_LINE_THICKNESS,
                        min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                        max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                        step_h = 5, step_v = 2,
                        rec_detect=False, show=False, test=False):
 
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    compos_all = []
    compos_rec = []
    compos_nonrec = []
    row, column = binary.shape[0], binary.shape[1]
    for i in range(0, row, step_h):
        for j in range(i % 2, column, step_v):
            if binary[i, j] == 255 and mask[i, j] == 0:
                # get connected area
                # region = util.boundary_bfs_connected_area(binary, i, j, mask)

                mask_copy = mask.copy()
                ff = cv2.floodFill(binary, mask, (j, i), None, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                if ff[0] < min_obj_area: continue
                mask_copy = mask - mask_copy
                region = np.reshape(cv2.findNonZero(mask_copy[1:-1, 1:-1]), (-1, 2))
                region = [(p[1], p[0]) for p in region]

                # filter out some compos
                component = Component(region, binary.shape)
                # calculate the boundary of the connected area
                # ignore small area
                if component.width <= 3 or component.height <= 3:
                    continue
                # check if it is line by checking the length of edges
                # if component.compo_is_line(line_thickness):
                #     continue

                if test:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary([component], binary.shape, show=True)

                compos_all.append(component)

                if rec_detect:
                    # rectangle check
                    if component.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                        component.rect_ = True
                        compos_rec.append(component)
                    else:
                        component.rect_ = False
                        compos_nonrec.append(component)
    if rec_detect:
        return compos_rec, compos_nonrec
    else:
        return compos_all
```


### Refinement of the blocks
#### Remove small area elements
``` python
def compo_filter(compos, min_area, img_shape):
    max_height = img_shape[0] * 0.8
    compos_new = []
    for compo in compos:
        if compo.area < min_area:
            continue
        if compo.height > max_height:
            continue
        ratio_h = compo.width / compo.height
        ratio_w = compo.height / compo.width
        if ratio_h > 50 or ratio_w > 40 or \
                (min(compo.height, compo.width) < 8 and max(ratio_h, ratio_w) > 10):
            continue
        compos_new.append(compo)
    return compos_new
```

#### Merge intersected elements
``` python 
def merge_intersected_compos(compos):
    changed = True
    while changed:
        changed = False
        temp_set = []
        for compo_a in compos:
            merged = False
            for compo_b in temp_set:
                if compo_a.compo_relation(compo_b) == 2:
                    compo_b.compo_merge(compo_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(compo_a)
        compos = temp_set.copy()
    return compos
```

#### NMS containment logic 
``` python
def bbox_relation_nms(self, bbox_b, bias=(0, 0)):
        '''
        Calculate the relation between two rectangles by nms
       :return: 
        -1 : a in b
         0  : a, b are not intersected
         1  : b in a
         2  : a, b are intersected
       '''
        col_min_a, row_min_a, col_max_a, row_max_a = self.put_bbox()
        col_min_b, row_min_b, col_max_b, row_max_b = bbox_b.put_bbox()

        bias_col, bias_row = bias
        # get the intersected area
        col_min_s = max(col_min_a - bias_col, col_min_b - bias_col)
        row_min_s = max(row_min_a - bias_row, row_min_b - bias_row)
        col_max_s = min(col_max_a + bias_col, col_max_b + bias_col)
        row_max_s = min(row_max_a + bias_row, row_max_b + bias_row)
        w = np.maximum(0, col_max_s - col_min_s)
        h = np.maximum(0, row_max_s - row_min_s)
        inter = w * h
        area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
        area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
        iou = inter / (area_a + area_b - inter)
        ioa = inter / self.box_area
        iob = inter / bbox_b.box_area

        if iou == 0 and ioa == 0 and iob == 0:
            return 0

        # contained by b
        if ioa >= 1:
            return -1
        # contains b
        if iob >= 1:
            return 1
        # not intersected with each other
        # intersected
        if iou >= 0.02 or iob > 0.2 or ioa > 0.2:
            return 2
        return 0

```
- biasness is added to remove any noise from the rectangular boxes in case of mobile UI or posters as well.

### Classifying elements
``` python
    if classifier is not None:
        classifier.predict([compo.compo_clipping(org) for compo in uicompos], uicompos)
``` 

### Text Detection OCR - vision api
``` python
def Google_OCR_makeImageData(imgpath):
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                # 'type': 'TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()

def ocr_detection_google(imgpath):
    start = time.process_time()
    url = 'https://vision.googleapis.com/v1/images:annotate'
    api_key = ''             # *** Replace with your own Key ***
    imgdata = Google_OCR_makeImageData(imgpath)
    response = requests.post(url,
                             data=imgdata,
                             params={'key': api_key},
                             headers={'Content_Type': 'application/json'})
    if 'responses' not in response.json():
        raise Exception(response.json())
    if response.json()['responses'] == [{}]:
        # No Text
        return None
    else:
        return response.json()['responses'][0]['textAnnotations'][1:]
```


### Text refinement code
``` python
# Filter noise like removing unwanted elements
def text_filter_noise(texts):
    valid_texts = []
    for text in texts:
        if len(text.content) <= 1 and text.content.lower() not in ['a', ',', '.', '!', '?', '$', '%', ':', '&', '+']:
            continue
        valid_texts.append(text)
    return valid_texts
```
Merge words into sentences after checking if they are on the same line

```python
def text_sentences_recognition(texts):
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_on_same_line(text_b, 'h', bias_justify=0.2 * min(text_a.height, text_b.height), bias_gap=2 * max(text_a.word_width, text_b.word_width)):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts

```

### Merging code
calc_intersection_area : calculate IOU, intersection over area A, intersection over Area B

#### Create Parent Child relationship
``` python
def refine_elements(compos, texts, intersection_bias=(2, 2), containment_ratio=0.8):
    '''
    1. remove compos contained in text
    2. remove compos containing text area that's too large
    3. store text in a compo if it's contained by the compo as the compo's text child element
    '''
    elements = []
    contained_texts = []
    for compo in compos:
        is_valid = True
        text_area = 0
        for text in texts:
            inter, iou, ioa, iob = compo.calc_intersection_area(text, bias=intersection_bias)
            if inter > 0:
                # the non-text is contained in the text compo
                if ioa >= containment_ratio:
                    is_valid = False
                    break
                text_area += inter
                # the text is contained in the non-text compo
                if iob >= containment_ratio and compo.category != 'Block':
                    contained_texts.append(text)
        if is_valid and text_area / compo.area < containment_ratio:
            for t in contained_texts:
                t.parent_id = compo.id
            compo.children += contained_texts
            elements.append(compo)

    # elements += texts
    for text in texts:
        if text not in contained_texts:
            elements.append(text)
    return elements
```

###  Accessibility Checks
#### Checking code
``` python
from wcag_contrast_ratio import rgb, ratio

class AccessibilityChecks:

    @staticmethod
    def contrast_checks(parent_ele, child_ele):

        text_color = child_ele.dominant_pixels
        bg_color = parent_ele.dominant_pixels
        contrast_ratio = ratio(text_color, bg_color)
        
        # Check for sufficient contrast
        if contrast_ratio < 4.5:
            return False
        else:
            return True
        
    @staticmethod
    def font_size_check(text_ele, heading=False):
        
        if not heading and text_ele.height < 16:
            return False
        else:
            return True
```

### Dominant pixel code
``` python
class PixelUtils:

    @staticmethod
    def getDominantPixelValue(roi, img):
        roi = roi
        return cv2.mean(img[roi['left']: roi['right'], roi['top'] : roi['bottom']])[:3]
```


### API Code
``` python

@app.route('/api/v1/process_image/', methods=['POST'])
@cross_origin()
def process_image():

    if 'image' not in request.files or 'ui_type' not in request.form:
        return jsonify({"error": "Missing 'file' or 'ui_type' parameter"}), 400
    
    file = request.files['image']
    ui_type = request.form['ui_type']
    
    # Save the file temporarily
    file_path = os.path.join('.', file.filename)
    file.save(file_path)
    response = process(input_path_img=file_path, ui_type=ui_type)

    # Optionally, return the processed image
    return jsonify(content={
            "message": "Image processed successfully",
            "ui_type": ui_type,
            "processed_image_details": response
        }), 200
```