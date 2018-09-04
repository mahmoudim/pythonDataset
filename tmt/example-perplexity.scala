// Stanford TMT Example 5 - Selecting LDA model parameters
// http://nlp.stanford.edu/software/tmt/0.4/

// tells Scala where to find the TMT classes
import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;
import java.io._;
import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

val source = CSVFile("dataset.csv") ~> IDColumn(1);

val tokenizer = {
  SimpleEnglishTokenizer() ~>            // tokenize on space and punctuation
  //CaseFolder() ~>                        // lowercase everything
  //WordsAndNumbersOnlyFilter() ~>         // ignore non-words and non-numbers
  MinimumLengthFilter(1)                 // take terms with >=3 characters
}

val text = {
  source ~>                              // read from the source file
  Column(2) ~>                           // select column containing text
  TokenizeWith(tokenizer) ~>             // tokenize with tokenizer above
  TermCounter() ~>                       // collect counts (needed below)
  TermMinimumDocumentCountFilter(1) ~>   // filter terms in <4 docs
  TermDynamicStopListFilter(0) ~>       // filter out 30 most common terms
  DocumentMinimumLengthFilter(1)         // take only docs with >=5 terms
}
// build a training dataset
val training = LDADataset(text);

// a list of pairs of (number of topics, perplexity)
var scores = List.empty[(Int,Double)];



// loop over various numbers of topics, training and evaluating each model
for (numTopics <- List(50,100,150,200,250,300,350,400,450,500,550,600,650,700)) {
  val params = LDAModelParams(numTopics = numTopics, dataset = training);
  val output = file("lda-"+training.signature+"-"+params.signature);

  val model = TrainCVB0LDA(params, training, output=output, maxIterations=1500);
  
  val perplexity = model.computePerplexity(training);


  val savestr = "res.txt"; 
  val f = new File(savestr);

  var out = new PrintWriter("null");
  if ( f.exists() && !f.isDirectory() ) {
      out = new PrintWriter(new FileOutputStream(new File(savestr), true));
  }
  else {
      out = new PrintWriter(savestr);
  }
  out.append("[perplexity] perplexity at "+numTopics+" topics: "+perplexity+"\n");
  out.close();

  scores :+= (numTopics, perplexity);
}