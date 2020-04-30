
# import joblib
# model.save_weights(checkpoint_dir + "/checkpoint.ckpt")
# joblib.dump(model, "model.pkl")
# joblib.dump(full_eval_image_generator, "full_eval_image_generator.pkl")
# joblib.dump(features, filename = "features.pkl")
# joblib.dump(filenames, filename = "filenames.pkl")
# joblib.dump(labels, filename = "labels.pkl")

# y_pred = model.predict(eval_image_generator)

# Z = model.evaluate(full_eval_image_generator, workers=4, use_multiprocessing=True)
# print(Z)

# print("===============================")
# print(y_pred)
# y_true = eval_image_generator.classes
# y_pred = np.argmax(y_pred, axis = 1)
# print(y_true, y_pred)
# score = quadratic_kappa(y_true, y_pred, 5)
# print(score)

# fmodel = load_features_model(model)

# F = fmodel.predict(eval_image_generator)
# # print(F)
# print(F.shape)