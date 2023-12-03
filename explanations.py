explanation_methods = ['Integrated_Gradients', 
                       'Input_X_Gradient', 
                       'SmoothGrad',
                       'Vanilla_Gradients', 
                       'Guided_Backprop',
                       'Occlusion',
                       'Lime', 
                       'KernelShap',
                       'DeepLift',
#                        'FeatureAblation',
#                        'FeaturePermutation',
#                        'ShapleyValueSampling',
          ]


#integrated gradients
def lem_ig(model, X):
    #model.zero_grad()
    attr_method = IntegratedGradients(model)
    X.requires_grad_()
    attributions = attr_method.attribute(X)
    return attributions

#input x gradient
def lem_ixg(model, X):
#     model.zero_grad()
    attr_method = InputXGradient(model)
    X.requires_grad_()
    attributions = attr_method.attribute(X)
    return attributions

#smoothgrad
def lem_sg(model, X):
#     model.zero_grad()
    attr_method = NoiseTunnel(Saliency(model))
    X.requires_grad_()
    attributions = attr_method.attribute(X, nt_type='smoothgrad', abs=False, stdevs=0.1)
    return attributions


#vanilla gradients
def lem_vg(model, X):
#     model.zero_grad()
    attr_method = Saliency(model)
    X.requires_grad_()
    attributions = attr_method.attribute(X, abs=False)
    return attributions

#guided backpropagation
def lem_gb(model, X):
#     model.zero_grad()
    attr_method = GuidedBackprop(model)
    X.requires_grad_()
    attributions = attr_method.attribute(X)
    return attributions


#occlusion
def lem_oc(model, X,):
    attr_method = Occlusion(model)
    attributions = attr_method.attribute(X, sliding_window_shapes=(1,))
    return attributions


#lime
def lem_lime(model, X):
    attr_method = Lime(model)
    attributions = attr_method.attribute(X, n_samples=100)
    return attributions


#kernelshap
def lem_ks(model, X):
    attr_method = KernelShap(model)
    attributions = attr_method.attribute(X, n_samples=100)
    return attributions

#deeplift
def lem_dpl(model, X):
    attr_method = KernelShap(model)
    attributions = attr_method.attribute(X)
    return attributions



explainer_attr_dict = {
    'Integrated_Gradients' : lem_ig, 
    'Input_X_Gradient' : lem_ixg, 
    'SmoothGrad' : lem_sg,
    'Vanilla_Gradients' : lem_vg, 
    'Guided_Backprop' : lem_gb,
    'Occlusion' : lem_oc,
    'Lime' : lem_lime,
    'KernelShap' : lem_ks,
    'DeepLift': lem_dpl,
}


NUM_EPOCHS = 20

for epoch in np.arange(1, NUM_EPOCHS+1):
    #get model
    model = models[epoch]
    print(epoch)
    directory_subdirectory = './data/xapi/NN/feature_importance/epoch_' + str(epoch)
    os.mkdir(directory_subdirectory)
    
    for method in explanation_methods:
        print(method)
        all_attributions = []
        for x in x_test_scaled.values:
            x = torch.from_numpy(x.astype(np.float32))
            x = Variable(torch.FloatTensor(x), requires_grad=True)
            x = torch.reshape(x, (1, x_test_scaled.shape[1]))
            func_explainer = explainer_attr_dict[method]
            attr_x = func_explainer(model, x)
            attr_x = attr_x.detach().numpy()
            all_attributions.append(attr_x[0])
        feature_importance = pd.DataFrame(all_attributions, columns=feature_names)
        file_name = method + '_epoch_' + str(epoch) + '.csv'
        feature_importance.to_csv('./data/xapi/NN/feature_importance/epoch_'+ str(epoch) +'/' + file_name, index=False)
