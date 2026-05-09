if st.button("Answer Query"):

    if not query:

        st.warning("Please enter a query")

    else:

        # RETRIEVE CONTEXT

        with st.spinner(
            "Searching Knowledge Base..."
        ):

            contexts = retrieve_context(
                query,
                local_kb
            )

        # GENERATE ANSWER

        with st.spinner(
            "Generating Answer..."
        ):

            answer = generate_answer(
                query,
                contexts
            )
        # DISPLAY OUTPUT

        st.subheader("Answer")

        st.write(answer)

        with st.expander(
            "Retrieved Context"
        ):

            for i, ctx in enumerate(
                contexts,
                1
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(ctx)
