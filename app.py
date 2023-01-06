from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pickle
from text_preprocess import make_prediction, getTfidfVec, prepareDataset, get_prediction_res_by_class

app = Flask(__name__, static_url_path='/static')

app.config["TEMPLATES_AUTO_RELOAD"] = True

model = None

@app.route("/")
def beranda():
	return render_template('index.html')

@app.route("/api/sentiment_analysis_text", methods=['POST'])
def sentimenAnalysisApiText():

	# get post data
	comment = request.form['comment']

	tfidfVec = getTfidfVec(prepareDataset()['clean_comment'])

	model = pickle.load(open('naivebayesclassifier.pkl', 'rb'))
	pred_res = make_prediction(tfidfVec, model, comment)

	if pred_res==0:
		pred_res = 'negative'
	else:
		pred_res = 'positive'

	return jsonify({
		"sentiment": pred_res,
	})

@app.route("/api/sentiment_analysis_brand", methods=['POST'])
def sentimenAnalysisApiBrand():

	# get post data
	brand = request.form['class']

	df = prepareDataset()
	negative_sentiment, positive_sentiment = get_prediction_res_by_class(df, brand)

	return jsonify({
		'negative_sentiment': '%.2f' % (negative_sentiment*100),
		'positive_sentiment': '%.2f' % (positive_sentiment*100)
	})


if __name__ == '__main__':

	# Run Flask di localhost 
	app.run(host="localhost", port=5555, debug=True)
	
	


