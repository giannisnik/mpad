from dataloader import DocumentGraphDataset
from learner import Learner
from parser_utils import get_args
from preprocess_docs import CorpusPreProcessor

if __name__ == "__main__":

    args, device = get_args()

    ######################################
    # Instantiate dataset
    ######################################
    corpus_prepper = CorpusPreProcessor(min_freq_word=1, multi_label=False)
    # Read data
    docs, labels, n_labels, word2idx = corpus_prepper.load_clean_corpus(
        args.path_to_dataset
    )
    # Split into train/val
    train_docs, dev_docs, train_labels, dev_labels = corpus_prepper.split_corpus(
        docs, labels, args.percentage_dev
    )
    # Load embeddings
    embeddings = corpus_prepper.load_embeddings(
        args.path_to_embeddings, word2idx, embedding_type="word2vec"
    )

    # Instantiate dataloader
    dataset_train = DocumentGraphDataset(
        docs=train_docs,
        labels=train_labels,
        word2idx=word2idx,
        window_size=args.window_size,
        use_master_node=args.use_master_node,
        normalize_edges=args.normalize,
        use_directed_edges=args.directed,
    )

    dataloader_train = dataset_train.to_dataloader(
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    dataset_dev = DocumentGraphDataset(
        docs=dev_docs,
        labels=dev_labels,
        word2idx=word2idx,
        window_size=args.window_size,
        use_master_node=args.use_master_node,
        normalize_edges=args.normalize,
        use_directed_edges=args.directed,
    )

    dataloader_dev = dataset_dev.to_dataloader(
        batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    ######################################
    # Initiate model
    ######################################

    learner = Learner(
        experiment_name=args.experiment_name, device=device, multi_label=False
    )
    learner.init_model(
        args.model_type,
        lr=args.lr,
        n_feat=embeddings.shape[1],
        n_message_passing=args.message_passing_layers,
        n_hid=args.hidden,
        n_penultimate=args.penultimate,
        n_class=n_labels,
        dropout=args.dropout,
        embeddings=embeddings,
        use_master_node=args.use_master_node,
    )

    ######################################
    # Start training
    ######################################

    eval_every = (
        len(dataloader_train) if args.eval_every == "epoch" else int(args.eval_every)
    )

    for epoch in range(args.epochs):

        learner.train_epoch(
            dataloader_train, eval_every=eval_every
        )

        learner.evaluate(dataloader_dev)

    ######################################
    # Infer Test Set
    ######################################
    if args.do_evaluate:

        print("Loading best model to infer test set...")
