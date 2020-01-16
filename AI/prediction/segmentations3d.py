from inout.readDataPreproc import read_preproc_imgs_and_ctrs_itk
from constants_proj.AI_proj_params import *
from img_viz.medical import MedicalImageVisualizer
import SimpleITK as sitk
from os.path import join



def proc_single_case_prostate(config, current_folder, dsc_data):
    '''
    This function process the segmentation of a single case. It makes the NN predition and saves the results
    :return:
    '''
    input_folder = config[ClassificationParams.input_folder]
    output_folder = config[ClassificationParams.output_folder]
    output_imgs_folder = config[ClassificationParams.output_imgs_folder]

    type_segmentation = config[ClassificationParams.segmentation_type]
    model = config[ModelParams.MODEL]
    # Reads original image and prostate
    if model == AiModels.UNET_3D_3_STREAMS:
        # All these names are predefined, for any other 3d segmentation we will need to create a different configuration
        if type_segmentation == SegmentationTypes.PROSTATE or type_segmentation == SegmentationTypes.PZ or type_segmentation == SegmentationTypes.TZ:
            img_names = ['img_tra.nrrd','hr_tra.nrrd','roi_tra.nrrd', 'roi_sag.nrrd', 'roi_cor.nrrd'],
            if type_segmentation == SegmentationTypes.PROSTATE:
                ctr_names = ['ctr_pro.nrrd', 'roi_ctr_pro.nrrd', 'hr_ctr_pro.nrrd', '']
            elif type_segmentation == SegmentationTypes.PZ:
                ctr_names = ['roi_ctr_pz.nrrd']
            elif type_segmentation == SegmentationTypes.TZ:
                ctr_names = ['roi_ctr_pro.nrrd','roi_ctr_pz.nrrd']
            else:
                raise Exception(F"This type os segmentation doesn't have a configuration. Model: {model} Type: {type_segmentation}")

            imgs_itk, ctrs_itk, sizeROI, startROI, _= read_preproc_imgs_and_ctrs_itk(input_folder,
                                                                                     folders_to_read=current_folder,
                                                                                     img_names=img_names,
                                                                                     ctr_names=ctr_names)
            [img_tra_original, img_tra_HR, ctr_pro, ctr_pro_HR, roi_ctr_pro, startROI, sizeROI] = \
                    else:
        raise Exception(F"Model doesn't have a segmentation configuration, please build one. Model: {model}")


    np_ctr_pro = sitk.GetArrayViewFromImage(ctr_pro)
    np_roi_ctr_pro = sitk.GetArrayViewFromImage(roi_ctr_pro)
    # Reads PZ and input for NN
    if type_segmentation == 'PZ' or type_segmentation == 'Prostate':
        [ctr_pz, ctr_pz_HR, roi_ctr_pz] = readPZ(input_folder, current_folder, multistream, img_size)
        np_ctr_pz = sitk.GetArrayViewFromImage(ctr_pz)
        np_roi_ctr_pz = sitk.GetArrayViewFromImage(roi_ctr_pz)
    # Reads Lesion and input for NN
    if type_segmentation == 'Lesion2D': # TODO not working
        [ctr_lesion, ctr_lesion_HR, roi_ctr_lesion] = readLesion(input_folder, current_folder)

    [roi_img1, roi_img2, roi_img3] = readROI(input_folder, current_folder, type_segmentation)

    print('Predicting image {} ({})....'.format(current_folder, input_folder))
    output_NN = makePrediction(model, roi_ctr_pro, img_size, roi_img1, roi_img2, roi_img3, multistream)

    # ************** Binary threshold and largest connected component ******************
    print('Threshold and largest component...')
    pred_nn = sitk.GetImageFromArray(output_NN[0,:,:,:,0])
    pred_nn = utils.binaryThresholdImage(pred_nn, threshold)
    pred_nn = utils.getLargestConnectedComponents(pred_nn)
    np_pred_nn = sitk.GetArrayViewFromImage(pred_nn)

    # ************** Compute metrics for ROI ******************
    c_img_folder = join(output_imgs_folder,current_folder)
    print('Metrics...')
    if type_segmentation == 'Prostate':
        cur_dsc_roi = numpy_dice(np_roi_ctr_pro, np_pred_nn)
        # cur_haus_roi = numpy_hausdorff(np_roi_ctr_pro, np_pred_nn)
        cur_haus_roi = -1
    if type_segmentation == 'PZ':
        cur_dsc_roi = numpy_dice(np_roi_ctr_pz, np_pred_nn)
        # cur_haus_roi = numpy_hausdorff(np_roi_ctr_pz, np_pred_nn)
        cur_haus_roi = -1
    print(F'--------------{c_img_folder} DSC ROI: {cur_dsc_roi:02.2f}  ------------')

    # ************** Visualize and save results for ROI ******************
    slices = roi_slices
    # title = F'{type_segmentation} {current_folder} DSC {cur_dsc_roi:02.2f} Hausdorff {cur_haus_roi:2.2f}'
    title = F'DSC {cur_dsc_roi:02.3f}'
    print('Making ROI images...')
    # if type_segmentation == 'Lesion':
    #     utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, subtitles=[title], contours=[roi_ctr_pro, roi_ctr_lesion, pred_nn],
    #                        savefig=join(output_imgs_folder,'ROI_LESION_'+current_folder), labels=['Prostate','GT','NN'])
    # if type_segmentation == 'Prostate':
    #     utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, subtitles=[title], contours=[roi_ctr_pro, pred_nn],
    #                         savefig=join(output_imgs_folder,'ROI_PROSTATE_'+current_folder), labels=['GT','NN'])
    # if type_segmentation == 'PZ':
    #     utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, subtitles=[title], contours=[roi_ctr_pro, roi_ctr_pz, pred_nn],
    #                          savefig=join(output_imgs_folder,'ROI_PZ_'+current_folder), labels=['Prostate','GT','NN'])

    # ************** Save ROI segmentation *****************
    if save_segmentations:
        print('Saving original prediction (ROI)...')
        if not os.path.exists(join(output_folder, current_folder)):
            os.makedirs(join(output_folder, current_folder))
        sitk.WriteImage(pred_nn, join(output_folder, current_folder, 'predicted_roi.nrrd'))

    dsc_data.loc[current_folder][metric_names.get('dsc_roi')] = cur_dsc_roi
    dsc_data.loc[current_folder][metric_names.get('hau_roi')] = cur_haus_roi

    # ************** Plot and save Metrics for ROI *****************

    # ************** Compute everything but for the original resolution *****************
    if not(compute_original_resolution): # in this case we do not upscale, just show the prediction
        print('Getting Original resolution ...')
        output_predicted_original = sitk.Image(img_tra_HR.GetSize(), sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(output_predicted_original) # Gets an array same size as original image
        arr[:] = 0 # Make everything = 0
        arr[startROI[2]:startROI[2]+sizeROI[2], startROI[1]:startROI[1]+sizeROI[1],startROI[0]:startROI[0]+sizeROI[0]] = output_NN[0,:,:,:,0]
        output_predicted = sitk.GetImageFromArray(arr)
        output_predicted = utils.binaryThresholdImage(output_predicted, threshold)
        output_predicted = utils.getLargestConnectedComponents(output_predicted)
        output_predicted = sitk.BinaryFillhole(output_predicted, fullyConnected=True)
        output_predicted.SetOrigin(img_tra_HR.GetOrigin())
        output_predicted.SetDirection(img_tra_HR.GetDirection())
        output_predicted.SetSpacing(img_tra_HR.GetSpacing())
        if save_segmentations:
            sitk.WriteImage(output_predicted, join(output_folder, current_folder, 'predicted_HR.nrrd'))

        # original transversal space (high slice thickness), transform perdiction with shape-based interpolation (via distance transformation)
        # segm_dis = sitk.SignedMaurerDistanceMap(output_predicted, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)
        # segm_dis = utils.resampleToReference(output_predicted, img_tra_original, sitk.sitkLinear, -1) # TODO don't know why it had -1 here
        segm_dis = utils.resampleToReference(output_predicted, img_tra_original, sitk.sitkNearestNeighbor, 0)
        thresholded = utils.binaryThresholdImage(segm_dis, threshold)
        np_pred_nn_orig = sitk.GetArrayFromImage(thresholded)
        if type_segmentation == 'Prostate':
            np_pred_nn_orig = utils.getLargestConnectedComponentsBySliceAndFillHoles(np_pred_nn_orig)
        thresholded = sitk.GetImageFromArray(np_pred_nn_orig)

        if save_segmentations:
            sitk.WriteImage(thresholded, join(output_folder, current_folder, 'predicted_transversal_space.nrrd'))

        if type_segmentation == 'Prostate':
            cur_dsc_original = numpy_dice(np_ctr_pro, np_pred_nn_orig)
            # cur_haus_original = numpy_hausdorff(np_ctr_pro, np_pred_nn_orig)
            cur_haus_original = -1
        if type_segmentation == 'PZ':
            cur_dsc_original = numpy_dice(np_ctr_pz, np_pred_nn_orig)
            cur_haus_original = numpy_dice(np_ctr_pz, np_pred_nn_orig)

        title = '{} DSC {:02.3f}'.format(current_folder, cur_dsc_original)
        slices = config['orig_slices']
        print('Making Original images ...')
        # if type_segmentation == 'Lesion':
        #     utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, ctr_pz, thresholded],
        #                            labels=['Prostate','GT','NN'], savefig=join(output_imgs_folder, current_folder))
        # if type_segmentation == 'PZ':
        #     utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, ctr_pz, thresholded],
        #                            labels=['Prostate','PZ','NN'], savefig=join(output_imgs_folder, current_folder))
        # if type_segmentation == 'Prostate':
        #     utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, thresholded],
        #                            labels=['GT','NN'], savefig=join(output_imgs_folder, current_folder))

    dsc_data.loc[current_folder][metric_names.get('dsc_orig')] = cur_dsc_original
    dsc_data.loc[current_folder][metric_names.get('hau_orig')] = cur_haus_original

    return dsc_data

def getProperFolders(input_folder, cases):
    '''Depending on the value of cases it reads the proper folders from the list of folders'''
    # *********** Define which cases are we going to perform the segmentation **********
    if isinstance(cases,str):
        if cases == 'all':
            examples = os.listdir(input_folder)
    else:
        examples = ['Case-{:04d}'.format(case) for case in cases]
    examples.sort()

    return examples
