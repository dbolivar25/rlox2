pub trait ResultExtensions<T, E> {
    fn pure(item: T) -> Result<T, E> {
        Ok(item)
    }
}
